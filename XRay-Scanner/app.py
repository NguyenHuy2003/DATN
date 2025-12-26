import streamlit as st
import time
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO
import os
import requests
import random

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

# --- 2. CẤU HÌNH FONT CHỮ (FIX LỖI CAN'T OPEN FILE) ---
def setup_vietnamese_font():
    """
    Cấu hình font tiếng Việt:
    1. Thử dùng Arial trên Windows (Nhanh nhất).
    2. Nếu không có (Linux/Cloud), tải Roboto về.
    """
    font_name = "VietnameseFont"
    
    # 1. Ưu tiên font hệ thống Windows (Arial)
    windows_font_path = "C:\\Windows\\Fonts\\arial.ttf"
    if os.path.exists(windows_font_path):
        try:
            pdfmetrics.registerFont(TTFont(font_name, windows_font_path))
            return font_name
        except Exception:
            pass # Nếu lỗi thì bỏ qua, chuyển sang tải Roboto

    # 2. Tải font Roboto (Cho Linux/Cloud hoặc fallback)
    font_filename = "Roboto-Regular.ttf"
    font_path = os.path.abspath(font_filename) # Lấy đường dẫn tuyệt đối để tránh lỗi path
    font_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
    
    # Kiểm tra nếu file tồn tại nhưng bị lỗi (0 byte) thì xóa đi tải lại
    if os.path.exists(font_path) and os.path.getsize(font_path) < 1000:
        try:
            os.remove(font_path)
        except:
            pass

    if not os.path.exists(font_path):
        with st.spinner("Đang tải font hỗ trợ tiếng Việt (Lần đầu chạy)..."):
            try:
                response = requests.get(font_url, timeout=10)
                if response.status_code == 200:
                    with open(font_path, "wb") as f:
                        f.write(response.content)
                else:
                    st.warning("Không tải được font Roboto. PDF sẽ dùng font mặc định (có thể lỗi dấu).")
                    return "Helvetica"
            except Exception as e:
                st.warning(f"Lỗi mạng khi tải font: {e}. PDF sẽ dùng font mặc định.")
                return "Helvetica"

    # Đăng ký font với ReportLab
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        return font_name
    except Exception as e:
        st.warning(f"Lỗi đăng ký font: {e}. PDF sẽ dùng font mặc định.")
        if os.path.exists(font_path):
            try: os.remove(font_path)
            except: pass
        return "Helvetica"

# Gọi hàm setup font ngay khi chạy app và lưu tên font vào biến toàn cục
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
    .body-part-tag { 
        display: inline-block; 
        background: rgba(0, 242, 255, 0.1); 
        border: 1px solid #00f2ff; 
        color: #00f2ff; 
        padding: 2px 8px; 
        border-radius: 4px; 
        font-family: 'Orbitron', sans-serif; 
        font-size: 0.8rem;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. CÁC HÀM XỬ LÝ (HELPER) ---

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def mock_ai_predict(image):
    """
    Giả lập AI:
    1. Phân loại vùng xương (Body Part Classification)
    2. Chẩn đoán bệnh (Diagnosis)
    """
    # Danh sách các vùng xương trong MURA
    BODY_PARTS = [
        "XR_ELBOW (Khuỷu tay)", 
        "XR_FINGER (Ngón tay)", 
        "XR_FOREARM (Cẳng tay)", 
        "XR_HAND (Bàn tay)", 
        "XR_HUMERUS (Xương cánh tay)", 
        "XR_SHOULDER (Vai)", 
        "XR_WRIST (Cổ tay)"
    ]
    
    # 1. Giả lập nhận diện vùng xương (Random)
    detected_part = random.choice(BODY_PARTS)
    
    # 2. Giả lập chẩn đoán (Random)
    is_abnormal = np.random.rand() > 0.6 
    img_array = np.array(image.convert('RGB'))
    h, w, _ = img_array.shape
    
    if is_abnormal:
        label = "PHÁT HIỆN TỔN THƯƠNG"
        confidence = np.random.uniform(0.85, 0.99)
        css_class = "status-abnormal"
        # Vẽ box đỏ giả lập
        start = (np.random.randint(w//4, w//2), np.random.randint(h//4, h//2))
        end = (start[0] + np.random.randint(50, 150), start[1] + np.random.randint(50, 150))
        cv2.rectangle(img_array, start, end, (255, 0, 0), 2) 
        cv2.putText(img_array, f"CONF: {confidence:.2f}", (start[0], start[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # Ghi tên vùng xương lên ảnh
        cv2.putText(img_array, detected_part.split(' ')[0], (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    else:
        label = "KHÔNG CÓ BẤT THƯỜNG"
        confidence = np.random.uniform(0.90, 0.99)
        css_class = "status-normal"
        cv2.putText(img_array, "NORMAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Ghi tên vùng xương lên ảnh
        cv2.putText(img_array, detected_part.split(' ')[0], (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return label, confidence, Image.fromarray(img_array), css_class, detected_part

# --- 5. HÀM TẠO PDF (Hỗ trợ Tiếng Việt & Tổng hợp) ---
def create_pdf(results_list, is_summary=False):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Hàm vẽ 1 trang báo cáo
    def draw_page(res, is_first_page=True):
        if not is_first_page: c.showPage()
        
        # Header
        c.setFont(FONT_NAME, 20)
        c.drawCentredString(width / 2, height - 50, "BÁO CÁO CHẨN ĐOÁN X-QUANG")
        c.setFont(FONT_NAME, 10)
        c.drawCentredString(width / 2, height - 70, f"File: {res['filename']}")
        c.line(50, height - 80, width - 50, height - 80)

        # Thông tin chi tiết
        c.setFont(FONT_NAME, 12)
        c.drawString(50, height - 120, f"Vùng chụp (Phát hiện): {res['body_part']}")
        
        # Kết quả text
        c.setFont(FONT_NAME, 14)
        if "BẤT THƯỜNG" in res['label'] or "PHÁT HIỆN" in res['label']:
            c.setFillColor(colors.red)
        else:
            c.setFillColor(colors.green)
            
        c.drawString(50, height - 150, f"Kết luận AI: {res['label']}")
        
        c.setFillColor(colors.black)
        c.drawString(50, height - 170, f"Độ tin cậy: {res['confidence']*100:.2f}%")
        
        # Lưu ảnh tạm để chèn vào PDF
        temp_orig = f"temp_orig_{res['id']}.png"
        res['original_image'].save(temp_orig)
        
        temp_res = f"temp_res_{res['id']}.png"
        res['result_image'].save(temp_res)
        
        # Vẽ ảnh vào PDF
        img_w = 200
        img_h = 200
        
        c.drawImage(temp_orig, 50, height - 420, width=img_w, height=img_h, preserveAspectRatio=True)
        c.drawString(100, height - 440, "Ảnh Gốc")
        
        c.drawImage(temp_res, 300, height - 420, width=img_w, height=img_h, preserveAspectRatio=True)
        c.drawString(350, height - 440, "Kết Quả AI")
        
        # Footer
        c.setFont(FONT_NAME, 8)
        c.setFillColor(colors.grey)
        c.drawCentredString(width / 2, 30, "X-Ray Sentinel AI System - Internal Use Only")
        
        try:
            os.remove(temp_orig)
            os.remove(temp_res)
        except:
            pass

    for i, res in enumerate(results_list):
        is_first = (i == 0)
        draw_page(res, is_first_page=is_first)

    c.save()
    return buffer.getvalue()

# --- 6. GIAO DIỆN CHÍNH ---
def main():
    # --- KHỞI TẠO SESSION STATE ---
    if 'results' not in st.session_state:
        st.session_state['results'] = []
    if 'processed' not in st.session_state:
        st.session_state['processed'] = False

    st.markdown('<div class="hud-header">X-RAY SENTINEL</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c2:
        with st.expander("⚙️ SYSTEM CONFIG", expanded=False):
            st.selectbox("MODEL", ["DenseNet-169", "ResNet-50"])
            st.radio("MODE", ["Fast Scan", "Deep Diagnostic"])

    with c1:
        uploaded_files = st.file_uploader("UPLOAD X-RAYS", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

        # Nút Phân tích
        if uploaded_files:
            if st.button("🚀 KÍCH HOẠT QUÉT (ACTIVATE)"):
                st.session_state['results'] = [] 
                st.session_state['processed'] = True 
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {i+1}/{len(uploaded_files)}...")
                    image = Image.open(uploaded_file)
                    # Gọi AI giả lập (nhận thêm detected_part)
                    label, confidence, res_img, css, detected_part = mock_ai_predict(image)
                    
                    st.session_state['results'].append({
                        'id': i,
                        'filename': uploaded_file.name,
                        'original_image': image,
                        'result_image': res_img,
                        'label': label,
                        'confidence': confidence,
                        'css': css,
                        'body_part': detected_part # Lưu vùng xương vào session
                    })
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Done!")
                time.sleep(0.5)
                st.rerun() 

    # --- HIỂN THỊ KẾT QUẢ ---
    if st.session_state['processed'] and st.session_state['results']:
        st.markdown("---")
        
        if len(st.session_state['results']) > 0:
            st.success(f"✅ Đã xử lý xong {len(st.session_state['results'])} ảnh.")
            summary_pdf = create_pdf(st.session_state['results'], is_summary=True)
            st.download_button(
                label="📥 TẢI BÁO CÁO TỔNG HỢP (.PDF)",
                data=summary_pdf,
                file_name="Tong_Hop_Ket_Qua_XQuang.pdf",
                mime="application/pdf",
                use_container_width=True
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
                st.caption("Ảnh gốc")
                st.markdown(f"""
                <div class="scan-container">
                    <div class="scan-line"></div>
                    <img src="data:image/png;base64,{image_to_base64(res['original_image'])}" width="100%">
                </div>
                """, unsafe_allow_html=True)
                
            with c_img_res:
                st.caption("Kết quả AI")
                st.image(res['result_image'], use_container_width=True)
                
            with c_info:
                st.markdown(f"""
                <div class="{res['css']}">
                    {res['label']}<br>
                    <span style="font-size:0.8rem">CONFIDENCE: {res['confidence']*100:.2f}%</span>
                </div>
                <div style="margin-top:10px; color:#aaa; font-size:0.9rem">
                    Vùng quét: <b style="color:#fff">{res['body_part']}</b>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                pdf_data = create_pdf([res], is_summary=False)
                st.download_button(
                    label=f"📄 Tải báo cáo lẻ",
                    data=pdf_data,
                    file_name=f"Report_{res['filename']}.pdf",
                    mime="application/pdf",
                    key=f"btn_{res['id']}"
                )
            
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()