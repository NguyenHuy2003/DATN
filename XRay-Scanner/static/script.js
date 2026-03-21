// xử lý khi người dùng chọn ảnh
document.getElementById('imageinput').addEventListener('change', function(e) {
    const files = e.target.files;
    const scanbtn = document.getElementById('scanbtn');
    const filelist = document.getElementById('filelist');
    
    if (files.length > 0) {
        scanbtn.classList.remove('hidden');
        filelist.classList.remove('hidden');
        filelist.innerHTML = `Đã chọn ${files.length} ảnh. Hệ thống sẵn sàng phân tích.`;
    } else {
        scanbtn.classList.add('hidden');
        filelist.classList.add('hidden');
    }
});

let globalresults = []; // mảng lưu kết quả

document.getElementById('scanbtn').addEventListener('click', async () => {
    const fileinput = document.getElementById('imageinput');
    const modelselect = document.getElementById('modelselect').value;
    const files = fileinput.files;

    if (files.length === 0) {
        alert("Vui lòng chọn ít nhất 1 ảnh x-quang");
        return;
    }

    const container = document.getElementById('resultscontainer');
    container.innerHTML = "";
    globalresults = [];
    document.getElementById('summaryaction').classList.add('hidden');
    
    const loadingdiv = document.getElementById('loading');
    const progresstext = document.getElementById('progresstext');
    loadingdiv.classList.remove('hidden');

    for (let i = 0; i < files.length; i++) {
        progresstext.innerText = `${i + 1}/${files.length}`;
        const file = files[i];
        const formdata = new FormData();
        formdata.append("file", file);
        formdata.append("model_type", modelselect);

        try {
            const response = await fetch('/predict', { method: 'POST', body: formdata });
            const data = await response.json();
            
            data.filename = file.name;
            globalresults.push(data);

            const cardhtml = `
            <div class="glass-panel">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 15px;">
                    <span style="color:#00f2ff; font-weight:600;">ID: ${file.name}</span>
                    <span class="body-part-tag">${data.body_part}</span>
                </div>
                
                <div class="result-grid">
                    <div class="img-col">
                        <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:8px;">Ảnh gốc</p>
                        <div class="scan-container">
                            <img src="data:image/png;base64,${data.original_b64}">
                            <div class="scan-line"></div>
                        </div>
                    </div>
                    <div class="img-col">
                        <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:8px;">Kết quả phân tích</p>
                        <img src="data:image/png;base64,${data.result_b64}">
                    </div>
                    <div class="info-col">
                        <div class="${data.css_class}">
                            <h3 style="margin:0 0 10px 0;">${data.label}</h3>
                            <p style="margin:0;">Độ tin cậy: ${(data.confidence * 100).toFixed(2)}%</p>
                        </div>
                        <button class="cyber-btn" style="margin-top: 15px;" onclick='downloadpdf(${JSON.stringify(data)})'>Tải báo cáo chi tiết</button>
                    </div>
                </div>
            </div>`;
            
            container.insertAdjacentHTML('beforeend', cardhtml);

        } catch (error) {
            console.error("Lỗi khi quét file: ", file.name, error);
            alert(`Lỗi khi xử lý file: ${file.name}`);
        }
    }

    loadingdiv.classList.add('hidden');
    if (globalresults.length > 0) {
        document.getElementById('summaryaction').classList.remove('hidden');
    }
});

async function downloadpdf(data) {
    requestpdf([data], `Bao_cao_${data.filename}.pdf`);
}

document.getElementById('downloadsummarybtn').addEventListener('click', () => {
    requestpdf(globalresults, `Tong_hop_xquang.pdf`);
});

async function requestpdf(resultsarray, outfilename) {
    try {
        const response = await fetch('/generate_pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(resultsarray)
        });
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = outfilename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        alert("Lỗi khi tạo file pdf");
    }
}