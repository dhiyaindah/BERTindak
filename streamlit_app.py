from transformers import BertModel
import streamlit as st
import torch
from models.model import AGMencoder
from utils.preprocessing import prepare_input
from huggingface_hub import hf_hub_download

# ============ Device ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Fungsi untuk Load CSS ======
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Panggil CSS
local_css("styles/style.css")

# ============ Inisialisasi Best Model Hierarchical BERT ============
MODEL_PATH = hf_hub_download(
    repo_id="dhiyaindah/bertindak-model",
    filename="best_model_1e5_03.pth"
)

hidden_size = 768
num_classes = 4
dropout_rate=0.3

model = AGMencoder(hidden_size=hidden_size, num_classes=num_classes, dropout_rate=dropout_rate).to(device)

# Muat state_dict dari model terbaik yang disimpan
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Header dengan gradient dan shadow yang menarik
st.markdown("""
    <div class="app-header">
        <h1 class="app-title">
            <span>⚖️</span>
            <span>BERTindak</span>
        </h1>
        <div class="header-divider"></div>
        <p class="app-subtitle">
            Aplikasi Klasifikasi Kategori Putusan Hukuman<br>
            <span class="gradient-text">Menggunakan Hierarchical BERT + BiLSTM-Attention Gated Mechanism</span>
        </p>
    </div>
""", unsafe_allow_html=True)

# Form dengan desain card modern
with st.form("form_prediksi"):
    st.markdown("""
        <div class="form-container">
            <h3 class="form-title">
                📋 Input Data Kasus
            </h3>
            <div class="input-container">
    """, unsafe_allow_html=True)
    
    riwayat_tuntutan = st.text_area(
        "Riwayat Tuntutan", 
        placeholder="Masukkan riwayat tuntutan kasus di sini..."
    )
    
    fakta = st.text_area(
        "Fakta", 
        placeholder="Jelaskan fakta-fakta yang terungkap dalam persidangan..."
    )
    
    fakta_hukum = st.text_area(
        "Fakta Hukum", 
        placeholder="Jelaskan fakta hukum yang relevan dengan kasus..."
    )
    
    pertimbangan_hukum = st.text_area(
        "Pertimbangan Hukum", 
        placeholder="Jelaskan pertimbangan hukum yang digunakan..."
    )
    
    st.markdown("</div>", unsafe_allow_html=True)  # Tutup container input fields
    
    # Tombol submit dengan styling yang menarik
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submitted = st.form_submit_button(
            "Prediksi Kategori Hukuman", 
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)  # Tutup card form

if submitted:
    if not riwayat_tuntutan or not fakta or not fakta_hukum or not pertimbangan_hukum:
        # Desain warning
        st.markdown("""
        <div class="warning-message">
            <div>
                <h4 class="warning-title">⚠️ Perhatian!</h4>
                <p class="warning-text">Semua input harus diisi terlebih dahulu sebelum prediksi!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        input_ids, attention_mask, token_type_ids = prepare_input(
            riwayat_tuntutan,
            fakta,
            fakta_hukum,
            pertimbangan_hukum
        )

        input_ids = input_ids.unsqueeze(0)        # (1, 1, seq_len)
        attention_mask = attention_mask.unsqueeze(0)  # (1, 1, seq_len)

        batch = (input_ids.to(device), attention_mask.to(device))

        with torch.no_grad():
            outputs = model(batch)
            predicted_class = torch.argmax(outputs, dim=1).item()

        # Mapping label ke kategori + warna
        label_map = {
            0: ("Mild", "#6ab04c", "ringan. Lama hukuman 0 – 479 hari"),       # hijau
            1: ("Moderate", "#2980b9", "sedang. Lama hukuman 480 – 1079  hari"),   # biru
            2: ("Heavy", "#e67e22", "berat. Lama hukuman 1080 – 1799 hari"),      # oranye
            3: ("Very Heavy", "#c0392b", "sangat berat. Lama hukuman 1800 – 8000 hari")  # merah
        }

        result, color, keterangan = label_map[predicted_class]
        
        # Animasi loading sebelum hasil muncul
        with st.spinner('Menganalisis data...'):
            import time
            time.sleep(1.5)  # Simulasi proses analisis

        # Card hasil yang lebih menarik
        st.markdown(f"""
        <div class="result-card" style="border-left: 5px solid {color}">
            <h3 class="result-title">Hasil Prediksi Kategori Hukuman</h3>
            <div class="result-content">
                <div class="result-category" style="color: {color}">
                    {result}
                </div>
                <div class="result-badge" style="background-color: {color}15; color: {color}">
                    Kategori {keterangan}
                </div>
            </div>
            <p class="result-footer">
                Hasil berdasarkan analisis model AI terhadap input yang diberikan
            </p>
        </div>
        """, unsafe_allow_html=True)
