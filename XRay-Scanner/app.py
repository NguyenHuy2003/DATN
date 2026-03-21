from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image
import io
import base64
import tensorflow as tf
import os

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

app = Flask(__name__)

# cấu hình font tiếng việt cho báo cáo pdf
def setup_vietnamese_font():
    font_name = "vietnamese_font"
    windows_font_path = "c:\\windows\\fonts\\arial.ttf"
    try:
        pdfmetrics.registerFont(TTFont(font_name, windows_font_path))
        return font_name
    except: 
        return "Helvetica"

font_name = setup_vietnamese_font()

# tải trực tiếp các mô hình ai vào bộ nhớ
print("Đang tải các mô hình ai...")
models_dict = {
    'general': tf.keras.models.load_model('models/best_mura_multitask.keras', compile=False),
    'shoulder': tf.keras.models.load_model('models/best_shoulder_model.keras', compile=False) 
}
print("Tải mô hình hoàn tất!")

# từ điển ánh xạ nhãn vùng xương
parts_map = {
    0: 'XR_Elbow (Khuỷu tay)', 1: 'XR_Finger (Ngón tay)', 2: 'XR_Forearm (Cẳng tay)', 
    3: 'XR_Hand (Bàn tay)', 4: 'XR_Humerus (Cánh tay)', 5: 'XR_Shoulder (Vai)', 6: 'XR_Wrist (Cổ tay)'
}

# các hàm tiền xử lý dữ liệu
def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="png")
    return base64.b64encode(buffered.getvalue()).decode()

def preprocess_image_for_model(pil_img):
    img = np.array(pil_img.convert('L'))  # chuyển sang ảnh xám
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

# thuật toán grad-cam tạo bản đồ nhiệt
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="densenet169"):
    try:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # 1. Tìm layer cần trích xuất (thường là DenseNet169)
        conv_layer = None
        try:
            conv_layer = model.get_layer(last_conv_layer_name)
        except:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    conv_layer = layer
                    break
                    
        if conv_layer is None:
            return np.zeros((img_array.shape[1], img_array.shape[2]))

        # 2. BẮT CÓC DỮ LIỆU (Monkey-Patching)
        # Bỏ qua hoàn toàn việc tạo grad_model gây lỗi của Keras 3
        captured_features = []
        original_call = conv_layer.call # Lưu lại hàm gốc của layer

        # Tạo một hàm giả mạo để hứng dữ liệu
        def call_wrapper(*args, **kwargs):
            out = original_call(*args, **kwargs) # Vẫn tính toán bình thường
            captured_features.append(out)        # Bắt cóc kết quả đưa ra ngoài
            return out

        # Tráo đổi hàm của layer bằng hàm giả mạo
        conv_layer.call = call_wrapper

        try:
            with tf.GradientTape() as tape:
                # Gọi mô hình gốc để kích hoạt hàm giả mạo và thu thập dữ liệu trung gian
                preds = model(img_tensor, training=False)
                
                if not captured_features:   
                    return np.zeros((img_array.shape[1], img_array.shape[2]))
                    
                conv_out = captured_features[0] # Dữ liệu trung gian đã bị bắt
                
                # Tương thích tự động: Phân biệt mô hình vai (1 đầu ra) và đa nhiệm (2 đầu ra)
                if isinstance(preds, list):
                    abnormality_preds = preds[0]
                else:
                    abnormality_preds = preds
                    
                if len(abnormality_preds.shape) > 1:
                    class_channel = abnormality_preds[:, 0]
                else:
                    class_channel = abnormality_preds[0]

            # 3. Tính đạo hàm từ dữ liệu đã bắt cóc
            grads = tape.gradient(class_channel, conv_out)
            
        finally:
            # Sau khi tính xong trả lại hàm gốc để không làm hỏng mô hình
            conv_layer.call = original_call

        if grads is None:
            return np.zeros((img_array.shape[1], img_array.shape[2]))

        # 4. Tính toán và chuẩn hóa bản đồ nhiệt
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        max_heat = tf.math.reduce_max(heatmap)
        if max_heat != 0:
            heatmap = heatmap / max_heat
            
        return heatmap.numpy()

    except Exception as e:
        print(f"Lỗi ngầm Grad-CAM: {e}")
        return np.zeros((img_array.shape[1], img_array.shape[2]))

# hàm phủ bản đồ nhiệt lên ảnh gốc
def apply_heatmap(original_img, heatmap, alpha=0.4):
    # nếu bản đồ nhiệt trống (do lỗi keras ngầm), giữ nguyên ảnh gốc
    if np.max(heatmap) == 0:
        return original_img
        
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # tạo bản đồ nhiệt jet (opencv xuất ra hệ màu bgr)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # chuyển từ hệ màu bgr của opencv sang rgb của web để không bị lộn màu
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # phủ màu lên ảnh gốc
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, original_img, 1 - alpha, 0)
    return superimposed_img

# định tuyến (routing) của web
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    model_type = request.form.get('model_type', 'general')
    
    selected_model = models_dict.get(model_type, models_dict['general'])

    image = Image.open(file.stream)
    tensor_img, preprocessed_img_uint8 = preprocess_image_for_model(image)
    
    preds = selected_model.predict(tensor_img, verbose=0)
    
    abnormality_pred = float(preds[0][0][0]) if isinstance(preds, list) else float(preds[0][0])
    
    if isinstance(preds, list):
        part_idx = int(np.argmax(preds[1][0]))
        detected_part = parts_map.get(part_idx, "Không xác định")
    else:
        detected_part = "XR_Xương vai" if model_type == 'shoulder' else "Không xác định"
        
    res_img_rgb = cv2.cvtColor(preprocessed_img_uint8, cv2.COLOR_GRAY2RGB)
    is_abnormal = abnormality_pred > 0.5
    
    if is_abnormal:
        label = "Phát hiện tổn thương"
        css_class = "status-abnormal"
        conf = abnormality_pred
        
        # ve ban do nhiet neu phat hien benh
        try:
            heatmap = make_gradcam_heatmap(tensor_img, selected_model, "densenet169")
            res_img_rgb = apply_heatmap(res_img_rgb, heatmap, alpha=0.4)
        except Exception as e:
            print(f"loi khi ve ban do nhiet: {e}")
            
        # xoa lenh cv2.putText tai day de anh duoc sach se
    else:
        label = "Không có bất thường"
        css_class = "status-normal"
        conf = 1.0 - abnormality_pred
        
        # xoa lenh cv2.putText tai day de anh duoc sach se

    return jsonify({
        "label": label, "confidence": conf, "css_class": css_class, 
        "body_part": detected_part,
        "original_b64": image_to_base64(image),
        "result_b64": image_to_base64(Image.fromarray(res_img_rgb))
    })

# api xuất file báo cáo pdf
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    results = request.json
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    for i, res in enumerate(results):
        if i > 0: 
            c.showPage()
        
        c.setFont(font_name, 20)
        c.drawCentredString(width / 2, height - 50, "Báo cáo chẩn đoán x-quang bằng AI")
        c.setFont(font_name, 10)
        
        # lay ten file an toan
        filename = res.get('filename', f'image_{i+1}.png')
        c.drawCentredString(width / 2, height - 70, f"file: {filename}")
        c.line(50, height - 80, width - 50, height - 80)

        c.setFont(font_name, 12)
        c.drawString(50, height - 120, f"Vùng chụp: {res.get('body_part', 'không xác định')}")
        
        c.setFont(font_name, 14)
        label = res.get('label', '')
        if "tổn thương" in label.lower() or "bất thường" in label.lower(): 
            c.setFillColor(colors.red)
        else: 
            c.setFillColor(colors.green)
        c.drawString(50, height - 150, f"Kết luận AI: {label}")
        
        c.setFillColor(colors.black)
        conf = res.get('confidence', 0)
        c.drawString(50, height - 170, f"Độ tin cậy: {conf*100:.2f}%")
        
        # doc truc tiep tu ram de tranh loi cache cua reportlab
        orig_img_data = base64.b64decode(res['original_b64'])
        res_img_data = base64.b64decode(res['result_b64'])
        
        orig_reader = ImageReader(io.BytesIO(orig_img_data))
        res_reader = ImageReader(io.BytesIO(res_img_data))
        
        # in truc tiep object imagereader len canvas
        c.drawImage(orig_reader, 50, height - 420, width=200, height=200, preserveAspectRatio=True)
        c.drawString(100, height - 440, "Ảnh gốc")
        
        c.drawImage(res_reader, 300, height - 420, width=200, height=200, preserveAspectRatio=True)
        c.drawString(350, height - 440, "Kết quả phân tích (grad-cam)")
        
    c.save()
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name='Bao_cao_tong_hop.pdf', mimetype='application/pdf')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)