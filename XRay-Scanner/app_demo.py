import streamlit as st
import time
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO
import os
import requests
import tensorflow as tf

# --- THƯ VIỆN PDF (ReportLab) ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="AI Assistant X-Ray Diagnosis",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- TẢI MÔ HÌNH KERAS (Chỉ tải 1 lần nhờ st.cache_resource) ---
@st.cache_resource
def load_xray_model():
    model_path = 'best_mura_multitask.keras'
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy file mô hình '{model_path}'. Vui lòng đặt file vào cùng thư mục với code.")
        return None
    
    with st.spinner("Đang khởi động mạng Neural (DenseNet169)..."):
        # Load mô hình, compile=False vì chúng ta chỉ cần dự đoán (Inference), không train nữa
        model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Gọi hàm load model ngay từ đầu
AI_MODEL = load_xray_model()

# --- 2. CẤU HÌNH FONT CHỮ ---
def setup_vietnamese_font():
    font_name = "VietnameseFont"
    windows_font_path = "C:\\Windows\\Fonts\\arial.ttf"
    if os.path.exists(windows_font_path):
        try:
            pdfmetrics.registerFont(TTFont(font_name, windows_font_path))
            return font_name
        except: pass

    font_filename = "Roboto-Regular.ttf"
    font_path = os.path.abspath(font_filename)
    font_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
    
    if os.path.exists(font_path) and os.path.getsize(font_path) < 1000:
        try: os.remove(font_path)
        except: pass

    if not os.path.exists(font_path):
        try:
            response = requests.get(font_url, timeout=10)
            if response.status_code == 200:
                with open(font_path, "wb") as f:
                    f.write(response.content)
            else: return "Helvetica"
        except: return "Helvetica"

    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        return font_name
    except: return "Helvetica"

FONT_NAME = setup_vietnamese_font()

# --- 3. CSS GIAO DIỆN ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Orbitron:wght@400;700&display=swap');
    
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .hud-header { font-family: 'Orbitron', sans-serif; font-size: 3rem; text-align: center; 
                  background: linear-gradient(90deg, #00f2ff, #0078ff); -webkit-background-clip: text; 
                  -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; letter-spacing: 2px; }
    .glass-panel { background: rgba(20, 25, 40, 0.7); border: 1px solid rgba(0, 242, 255, 0.1);
                   border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    .status-normal { color: #10b981; font-weight: bold; border: 1px solid #10b981; padding: 5px; border-radius: 5px; text-align: center;}
    .status-abnormal { color: #ef4444; font-weight: bold; border: 1px solid #ef4444; padding: 5px; border-radius: 5px; text-align: center;}
    .stButton > button { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
                         border: 1px solid #334155; color: #00f2ff; font-family: 'Orbitron', sans-serif; }
    .scan-container { position: relative; border-radius: 8px; overflow: hidden; border: 1px solid #334155; }
    .scan-line { position: absolute; top: 0; left: 0; width: 100%; height: 2px; background: #00f2ff; 
                 box-shadow: 0 0 10px #00f2ff; animation: scan 2s infinite; opacity: 0.7; }
    @keyframes scan { 0% {top:0%} 50% {top:100%} 100% {top:0%} }
    .body-part-tag { display: inline-block; background: rgba(0, 242, 255, 0.1); border: 1px solid #00f2ff; 
                     color: #00f2ff; padding: 2px 8px; border-radius: 4px; font-family: 'Orbitron', sans-serif; 
                     font-size: 0.8rem; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 4. HÀM XỬ LÝ (TIỀN XỬ LÝ & DỰ ĐOÁN) ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def preprocess_image_for_model(pil_img):
    """ Tái tạo 100% bước tiền xử lý như lúc Training (CLAHE + Resize Padding) """
    # 1. Chuyển sang Grayscale NumPy Array
    img = np.array(pil_img.convert('L'))
    
    # 2. Cân bằng sáng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    # 3. Resize giữ tỷ lệ và thêm Padding đen
    target_size = (320, 320)
    h, w = img_clahe.shape
    r = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * r), int(h * r)
    resized_image = cv2.resize(img_clahe, (new_w, new_h))
    
    padded_image = np.zeros(target_size, dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
    
    # 4. Normalize về [0, 1] cho Neural Network
    tensor_img = padded_image.astype(np.float32) / 255.0
    tensor_img = np.expand_dims(tensor_img, axis=-1) # (320, 320, 1)
    tensor_img = np.expand_dims(tensor_img, axis=0)  # (1, 320, 320, 1) Batch size = 1
    
    return tensor_img, padded_image

def real_ai_predict(image, model):
    """ Chạy dự đoán bằng Keras Model thực tế """
    # Từ điển ánh xạ
    PARTS_MAP = {
        0: 'XR_ELBOW (Khuỷu tay)', 1: 'XR_FINGER (Ngón tay)', 2: 'XR_FOREARM (Cẳng tay)', 
        3: 'XR_HAND (Bàn tay)', 4: 'XR_HUMERUS (Xương cánh tay)', 5: 'XR_SHOULDER (Vai)', 6: 'XR_WRIST (Cổ tay)'
    }
    
    if model is None:
        return "LỖI MÔ HÌNH", 0.0, image, "status-abnormal", "Không xác định"
        
    # Tiền xử lý
    tensor_img, preprocessed_img_uint8 = preprocess_image_for_model(image)
    
    # Đưa vào mạng nơ-ron
    # Model output trả về list: [out_abnormality, out_part]
    preds = model.predict(tensor_img, verbose=0)
    abnormality_pred = preds[0][0][0] # Xác suất bị bệnh (Sigmoid)
    part_pred = preds[1][0]           # Xác suất 7 loại xương (Softmax)
    
    # Xử lý kết quả xương
    part_idx = np.argmax(part_pred)
    detected_part = PARTS_MAP.get(part_idx, "Không xác định")
    
    # Xử lý kết quả bệnh (Ngưỡng Threshold = 0.5)
    is_abnormal = abnormality_pred > 0.5
    if is_abnormal:
        confidence = abnormality_pred
        label = "PHÁT HIỆN TỔN THƯƠNG"
        css_class = "status-abnormal"
        color = (255, 0, 0) # Đỏ
        text_label = f"ABNORMAL: {confidence*100:.1f}%"
    else:
        confidence = 1.0 - abnormality_pred # Độ tự tin của việc KHÔNG có bệnh
        label = "KHÔNG CÓ BẤT THƯỜNG"
        css_class = "status-normal"
        color = (0, 255, 0) # Xanh lá
        text_label = f"NORMAL: {confidence*100:.1f}%"
        
    # Tạo ảnh hiển thị kết quả (Chuyển ảnh tiền xử lý sang RGB để vẽ màu)
    res_img_rgb = cv2.cvtColor(preprocessed_img_uint8, cv2.COLOR_GRAY2RGB)
    h, w, _ = res_img_rgb.shape
    
    # Ghi chú lên ảnh (Vì model phân loại không sinh ra tọa độ Bounding Box, ta chỉ ghi text overlay)
    cv2.putText(res_img_rgb, text_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(res_img_rgb, f"Part: {detected_part.split(' ')[0]}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return label, float(confidence), Image.fromarray(res_img_rgb), css_class, detected_part

# --- 5. HÀM TẠO PDF (Giữ nguyên) ---
def create_pdf(results_list, is_summary=False):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    def draw_page(res, is_first_page=True):
        if not is_first_page: c.showPage()
        
        c.setFont(FONT_NAME, 20)
        c.drawCentredString(width / 2, height - 50, "BÁO CÁO CHẨN ĐOÁN X-QUANG")
        c.setFont(FONT_NAME, 10)
        c.drawCentredString(width / 2, height - 70, f"File: {res['filename']}")
        c.line(50, height - 80, width - 50, height - 80)

        c.setFont(FONT_NAME, 12)
        c.drawString(50, height - 120, f"Vùng chụp (Phát hiện): {res['body_part']}")
        
        c.setFont(FONT_NAME, 14)
        if "BẤT THƯỜNG" in res['label'] or "PHÁT HIỆN" in res['label']: c.setFillColor(colors.red)
        else: c.setFillColor(colors.green)
            
        c.drawString(50, height - 150, f"Kết luận AI: {res['label']}")
        
        c.setFillColor(colors.black)
        c.drawString(50, height - 170, f"Độ tin cậy: {res['confidence']*100:.2f}%")
        
        temp_orig = f"temp_orig_{res['id']}.png"
        res['original_image'].save(temp_orig)
        
        temp_res = f"temp_res_{res['id']}.png"
        res['result_image'].save(temp_res)
        
        img_w, img_h = 200, 200
        c.drawImage(temp_orig, 50, height - 420, width=img_w, height=img_h, preserveAspectRatio=True)
        c.drawString(100, height - 440, "Ảnh Gốc")
        
        c.drawImage(temp_res, 300, height - 420, width=img_w, height=img_h, preserveAspectRatio=True)
        c.drawString(350, height - 440, "Kết Quả AI")
        
        c.setFont(FONT_NAME, 8)
        c.setFillColor(colors.grey)
        c.drawCentredString(width / 2, 30, "X-Ray Sentinel AI System - Internal Use Only")
        
        try:
            os.remove(temp_orig)
            os.remove(temp_res)
        except: pass

    for i, res in enumerate(results_list):
        draw_page(res, is_first_page=(i == 0))

    c.save()
    return buffer.getvalue()

# --- 6. GIAO DIỆN CHÍNH ---
def main():
    if 'results' not in st.session_state: st.session_state['results'] = []
    if 'processed' not in st.session_state: st.session_state['processed'] = False

    st.markdown('<div class="hud-header">X-RAY SENTINEL</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c2:
        with st.expander("⚙️ SYSTEM CONFIG", expanded=False):
            st.selectbox("MODEL", ["DenseNet-169 Multi-task"])
            st.radio("MODE", ["Deep Diagnostic"])

    with c1:
        uploaded_files = st.file_uploader("UPLOAD X-RAYS", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

        if uploaded_files:
            if st.button("🚀 KÍCH HOẠT QUÉT (ACTIVATE)"):
                st.session_state['results'] = [] 
                st.session_state['processed'] = True 
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Đang phân tích ảnh {i+1}/{len(uploaded_files)} bằng AI...")
                    image = Image.open(uploaded_file)
                    
                    # Gọi hàm dự đoán REAL AI
                    label, confidence, res_img, css, detected_part = real_ai_predict(image, AI_MODEL)
                    
                    st.session_state['results'].append({
                        'id': i, 'filename': uploaded_file.name, 'original_image': image,
                        'result_image': res_img, 'label': label, 'confidence': confidence,
                        'css': css, 'body_part': detected_part
                    })
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Phân tích hoàn tất!")
                time.sleep(0.5)
                st.rerun() 

    # --- HIỂN THỊ KẾT QUẢ ---
    if st.session_state['processed'] and st.session_state['results']:
        st.markdown("---")
        
        if len(st.session_state['results']) > 0:
            st.success(f"✅ Đã phân tích xong {len(st.session_state['results'])} ảnh X-Quang.")
            summary_pdf = create_pdf(st.session_state['results'], is_summary=True)
            st.download_button(
                label="📥 TẢI BÁO CÁO TỔNG HỢP (.PDF)", data=summary_pdf,
                file_name="Tong_Hop_Ket_Qua_XQuang.pdf", mime="application/pdf", use_container_width=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

        for res in st.session_state['results']:
            st.markdown(f"""
            <div class="glass-panel">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="color:#00f2ff; font-family:'Orbitron'">ID: {res['filename']}</span>
                    <span class="body-part-tag">{res['body_part']}</span>
                </div>
            """, unsafe_allow_html=True)
            
            c_img_orig, c_img_res, c_info = st.columns([1, 1, 1])
            
            with c_img_orig:
                st.caption("Ảnh gốc tải lên")
                st.markdown(f"""
                <div class="scan-container">
                    <div class="scan-line"></div>
                    <img src="data:image/png;base64,{image_to_base64(res['original_image'])}" width="100%">
                </div>
                """, unsafe_allow_html=True)
                
            with c_img_res:
                st.caption("Ảnh qua tiền xử lý (Model Input)")
                st.image(res['result_image'], use_container_width=True)
                
            with c_info:
                st.markdown(f"""
                <div class="{res['css']}">
                    {res['label']}<br>
                    <span style="font-size:0.8rem">ĐỘ TIN CẬY: {res['confidence']*100:.2f}%</span>
                </div>
                <div style="margin-top:10px; color:#aaa; font-size:0.9rem">
                    Vị trí xương: <b style="color:#fff">{res['body_part']}</b>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                pdf_data = create_pdf([res], is_summary=False)
                st.download_button(
                    label=f"📄 Xuất báo cáo y tế lẻ", data=pdf_data,
                    file_name=f"Report_{res['filename']}.pdf", mime="application/pdf", key=f"btn_{res['id']}"
                )
            
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()