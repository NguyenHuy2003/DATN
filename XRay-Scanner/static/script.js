let globalResults = []; // Lưu trữ toàn bộ kết quả để in PDF tổng hợp

document.getElementById('scanBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageInput');
    const modelSelect = document.getElementById('modelSelect').value;
    const files = fileInput.files;

    if (files.length === 0) {
        alert("Vui lòng chọn ít nhất 1 ảnh X-Quang!");
        return;
    }

    // Reset giao diện
    const container = document.getElementById('resultsContainer');
    container.innerHTML = "";
    globalResults = [];
    document.getElementById('summaryAction').classList.add('hidden');
    
    const loadingDiv = document.getElementById('loading');
    const progressText = document.getElementById('progressText');
    loadingDiv.classList.remove('hidden');

    // Lặp qua từng file để xử lý
    for (let i = 0; i < files.length; i++) {
        progressText.innerText = `${i + 1}/${files.length}`;
        const file = files[i];
        const formData = new FormData();
        formData.append("file", file);
        formData.append("model_type", modelSelect); // Gửi loại model xuống backend

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            
            // Lưu kết quả vào mảng toàn cục
            data.filename = file.name;
            globalResults.push(data);

            // Tạo khung HTML (Card) cho từng ảnh
            const cardHTML = `
            <div class="result-section glass-panel">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 15px;">
                    <span style="color:#00f2ff; font-family:'Orbitron'">ID: ${file.name}</span>
                    <span class="body-part-tag">${data.body_part}</span>
                </div>
                
                <div class="result-grid">
                    <div class="img-col">
                        <p>Ảnh gốc</p>
                        <div class="scan-container">
                            <img src="data:image/png;base64,${data.original_b64}" style="max-width: 100%; border-radius: 8px;">
                        </div>
                    </div>
                    <div class="img-col">
                        <p>Kết quả AI</p>
                        <img src="data:image/png;base64,${data.result_b64}" style="max-width: 100%; border-radius: 8px;">
                    </div>
                    <div class="info-col">
                        <div class="${data.css_class}" style="padding: 15px;">
                            <h3>${data.label}</h3>
                            <p>ĐỘ TIN CẬY: ${(data.confidence * 100).toFixed(2)}%</p>
                        </div>
                        <button class="cyber-btn" style="margin-top: 15px;" onclick='downloadPDF(${JSON.stringify(data)})'>📄 Tải báo cáo lẻ</button>
                    </div>
                </div>
            </div>`;
            
            // Gắn vào giao diện
            container.insertAdjacentHTML('beforeend', cardHTML);

        } catch (error) {
            console.error("Lỗi khi quét file: ", file.name, error);
            alert(`Lỗi khi xử lý file: ${file.name}`);
        }
    }

    loadingDiv.classList.add('hidden');
    // Hiện nút tải báo cáo tổng hợp
    if (globalResults.length > 0) {
        document.getElementById('summaryAction').classList.remove('hidden');
    }
});

// Hàm gọi API tạo PDF lẻ
async function downloadPDF(data) {
    requestPDF([data], `Report_${data.filename}.pdf`);
}

// Hàm gọi API tạo PDF tổng hợp
document.getElementById('downloadSummaryBtn').addEventListener('click', () => {
    requestPDF(globalResults, `Tong_hop_XQuang.pdf`);
});

// Logic gửi request in PDF
async function requestPDF(resultsArray, outFilename) {
    try {
        const response = await fetch('/generate_pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(resultsArray)
        });
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = outFilename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        alert("Lỗi khi tạo file PDF!");
    }
}