from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image
import io
import base64
import tensorflow as tf
import os
import requests

# Thư viện PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

app = Flask(__name__)

# --- CẤU HÌNH FONT TIẾNG VIỆT CHO PDF ---
def setup_vietnamese_font():
    font_name = "VietnameseFont"
    windows_font_path = "C:\\Windows\\Fonts\\arial.ttf"
    if os.path.exists(windows_font_path):
        try:
            pdfmetrics.registerFont(TTFont(font_name, windows_font_path))
            return font_name
        except: pass
    return "Helvetica"

FONT_NAME = setup_vietnamese_font()

# --- LOAD CÙNG LÚC 2 MÔ HÌNH ---
print("⏳ Đang tải các mô hình AI...")
MODELS = {}
try:
    MODELS['general'] = tf.keras.models.load_model('best_mura_multitask.keras', compile=False)
    print("✅ Đã tải Mô hình Tổng hợp.")
except:
    print("❌ Lỗi: Không tìm thấy best_mura_multitask.keras")

try:
    # BẠN ĐỔI TÊN FILE NÀY THÀNH TÊN MODEL XƯƠNG VAI CỦA BẠN NHÉ!
    MODELS['shoulder'] = tf.keras.models.load_model('best_shoulder_model.keras', compile=False)
    print("✅ Đã tải Mô hình Xương Vai.")
except:
    print("⚠️ Cảnh báo: Chưa có file best_shoulder_model.keras, dùng tạm mô hình tổng hợp.")
    MODELS['shoulder'] = MODELS.get('general')

PARTS_MAP = {
    0: 'XR_ELBOW (Khuỷu tay)', 1: 'XR_FINGER (Ngón tay)', 2: 'XR_FOREARM (Cẳng tay)', 
    3: 'XR_HAND (Bàn tay)', 4: 'XR_HUMERUS (Xương cánh tay)', 5: 'XR_SHOULDER (Vai)', 6: 'XR_WRIST (Cổ tay)'
}

# (GIỮ NGUYÊN HÀM preprocess_image_for_model VÀ image_to_base64 TỪ PHIÊN BẢN TRƯỚC)
def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def preprocess_image_for_model(pil_img):
    img = np.array(pil_img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    target_size = (320, 320)
    h, w = img_clahe.shape
    r = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img_clahe, (new_w, new_h))
    padded = np.zeros(target_size, dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    tensor_img = padded.astype(np.float32) / 255.0
    tensor_img = np.expand_dims(tensor_img, axis=-1)
    tensor_img = np.expand_dims(tensor_img, axis=0)
    return tensor_img, padded

# --- ENDPOINTS ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    model_type = request.form.get('model_type', 'general') # Lấy model từ JS gửi lên
    
    selected_model = MODELS.get(model_type, MODELS['general'])

    image = Image.open(file.stream)
    tensor_img, preprocessed_img_uint8 = preprocess_image_for_model(image)
    
    preds = selected_model.predict(tensor_img, verbose=0)
    
    # Xử lý Logic Multi-task (Nếu Model xương vai của bạn chỉ có 1 đầu ra, bạn phải viết thêm câu if ở đây)
    abnormality_pred = float(preds[0][0][0]) if isinstance(preds, list) else float(preds[0][0])
    
    if isinstance(preds, list):
        part_idx = int(np.argmax(preds[1][0]))
        detected_part = PARTS_MAP.get(part_idx, "Không xác định")
    else:
        detected_part = "XR_SHOULDER (Chuyên biệt)" if model_type == 'shoulder' else "Unknown"
        
    res_img_rgb = cv2.cvtColor(preprocessed_img_uint8, cv2.COLOR_GRAY2RGB)
    if abnormality_pred > 0.5:
        label, css_class, conf = "PHÁT HIỆN TỔN THƯƠNG", "status-abnormal", abnormality_pred
        cv2.putText(res_img_rgb, f"ABNORMAL: {conf*100:.1f}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    else:
        label, css_class, conf = "KHÔNG CÓ BẤT THƯỜNG", "status-normal", 1.0 - abnormality_pred
        cv2.putText(res_img_rgb, f"NORMAL: {conf*100:.1f}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return jsonify({
        "label": label, "confidence": conf, "css_class": css_class, 
        "body_part": detected_part,
        "original_b64": image_to_base64(image),
        "result_b64": image_to_base64(Image.fromarray(res_img_rgb))
    })

# --- HÀM TẠO PDF ---
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    results = request.json
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    for i, res in enumerate(results):
        if i > 0: c.showPage()
        
        c.setFont(FONT_NAME, 20)
        c.drawCentredString(width / 2, height - 50, "BÁO CÁO CHẨN ĐOÁN X-QUANG")
        c.setFont(FONT_NAME, 10)
        c.drawCentredString(width / 2, height - 70, f"File: {res['filename']}")
        c.line(50, height - 80, width - 50, height - 80)

        c.setFont(FONT_NAME, 12)
        c.drawString(50, height - 120, f"Vùng chụp: {res['body_part']}")
        
        c.setFont(FONT_NAME, 14)
        if "TỔN THƯƠNG" in res['label']: c.setFillColor(colors.red)
        else: c.setFillColor(colors.green)
        c.drawString(50, height - 150, f"Kết luận AI: {res['label']}")
        
        c.setFillColor(colors.black)
        c.drawString(50, height - 170, f"Độ tin cậy: {res['confidence']*100:.2f}%")
        
        # Giải mã ảnh từ base64 lưu tạm ra ổ cứng để vẽ vào PDF
        orig_img_data = base64.b64decode(res['original_b64'])
        res_img_data = base64.b64decode(res['result_b64'])
        
        with open("temp_orig.png", "wb") as f: f.write(orig_img_data)
        with open("temp_res.png", "wb") as f: f.write(res_img_data)
        
        c.drawImage("temp_orig.png", 50, height - 420, width=200, height=200, preserveAspectRatio=True)
        c.drawString(100, height - 440, "Ảnh Gốc")
        c.drawImage("temp_res.png", 300, height - 420, width=200, height=200, preserveAspectRatio=True)
        c.drawString(350, height - 440, "Kết Quả AI")
        
    c.save()
    buffer.seek(0)
    
    # Xóa file tạm
    try:
        os.remove("temp_orig.png")
        os.remove("temp_res.png")
    except: pass
    
    return send_file(buffer, as_attachment=True, download_name='Bao_Cao_XQuang.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)