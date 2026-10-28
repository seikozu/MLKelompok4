import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Customer Churn",
    layout="wide"
)

# ============================================================
# LOAD PARAMETER & EVALUASI
# ============================================================
@st.cache_resource
def load_assets():
    label_encoders = joblib.load("label_encoders.pkl")
    normalisasi_params = joblib.load("normalisasi_params.pkl")
    fitur_klasifikasi = joblib.load("fitur_klasifikasi.pkl")
    kolom_info = joblib.load("kolom_info.pkl")
    kolom_numerik = joblib.load("kolom_numerik.pkl")
    hasil_evaluasi = joblib.load("hasil_evaluasi.pkl")
    
    # Load ketiga model
    models = {
        "SVM": joblib.load("model_svm.pkl"),
        "Naive Bayes": joblib.load("model_naive_bayes.pkl"),
        "Decision Tree": joblib.load("model_decision_tree.pkl")
    }
    return label_encoders, normalisasi_params, fitur_klasifikasi, kolom_info, kolom_numerik, hasil_evaluasi, models

try:
    label_encoders, normalisasi_params, fitur_klasifikasi, kolom_info, kolom_numerik, hasil_evaluasi, models = load_assets()
except Exception as e:
    st.error(f"Gagal memuat file .pkl: {e}. Pastikan file .pkl sudah di-generate!")
    st.stop()

# ============================================================
# SIDEBAR: DROPDOWN PEMILIHAN MODEL
# ============================================================
st.sidebar.header("Pengaturan Model")

# Dropdown untuk memilih model
pilihan_model = st.sidebar.selectbox(
    label="Pilih Model Klasifikasi:",
    options=["SVM", "Naive Bayes", "Decision Tree"],
    index=0  # Default ke SVM (Akurasi tertinggi)
)

# Ambil model terpilih
model = models[pilihan_model]

# Ambil akurasi model terpilih dari dictionary hasil_evaluasi
mapping_eval_key = {
    "SVM": "svm",
    "Naive Bayes": "naive_bayes",
    "Decision Tree": "decision_tree"
}
eval_key = mapping_eval_key[pilihan_model]
akurasi_terpilih = hasil_evaluasi[eval_key]["accuracy"]

# Tampilkan akurasi model yang sedang dipilih
st.sidebar.info(f"**Model Aktif:** {pilihan_model}")
st.sidebar.metric("Akurasi Model", f"{akurasi_terpilih * 100:.2f}%")

# Ringkasan Evaluasi Semua Model
st.sidebar.divider()
st.sidebar.subheader("Perbandingan Akurasi")
st.sidebar.text(f"SVM           : {hasil_evaluasi['svm']['accuracy']*100:.2f}% (Terbaik)")
st.sidebar.text(f"Naive Bayes   : {hasil_evaluasi['naive_bayes']['accuracy']*100:.2f}%")
st.sidebar.text(f"Decision Tree : {hasil_evaluasi['decision_tree']['accuracy']*100:.2f}%")

# ============================================================
# HEADER APLIKASI
# ============================================================
st.title("Sistem Prediksi Telco Customer Churn")
st.caption(f"Menggunakan Model Klasifikasi: **{pilihan_model}**")
st.divider()

# ============================================================
# FORM INPUT USER
# ============================================================
st.subheader("Form Input Data Pelanggan")

input_data = {}
col1, col2 = st.columns(2)

for i, fitur in enumerate(fitur_klasifikasi):
    target_col = col1 if i % 2 == 0 else col2
    
    with target_col:
        if fitur in kolom_numerik:
            min_val = float(normalisasi_params[fitur]['min'])
            max_val = float(normalisasi_params[fitur]['max'])
            default_val = float((min_val + max_val) / 2)
            
            input_data[fitur] = st.number_input(
                label=f"Nilai {fitur}",
                min_value=0.0,
                max_value=max_val * 2,
                value=default_val,
                help=f"Rentang data training: {min_val:.2f} - {max_val:.2f}"
            )
        else:
            opsi = kolom_info.get(fitur, [])
            if not opsi and fitur in label_encoders:
                opsi = list(label_encoders[fitur].classes_)
            
            input_data[fitur] = st.selectbox(
                label=f"Pilihan {fitur}",
                options=opsi
            )

st.divider()

# ============================================================
# PROSES PREDIKSI
# ============================================================
if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
    df_input = pd.DataFrame([input_data])
    df_processed = df_input.copy()
    
    # Preprocessing & Normalisasi
    for col in df_processed.columns:
        if col in kolom_numerik:
            t_min = normalisasi_params[col]['min']
            t_max = normalisasi_params[col]['max']
            df_processed[col] = (df_processed[col] - t_min) / (t_max - t_min)
        else:
            le = label_encoders[col]
            df_processed[col] = le.transform(df_processed[col])
            
    df_processed = df_processed[fitur_klasifikasi]
    prediksi = model.predict(df_processed)[0]
    
    # Cek probabilitas (jika model mendukung)
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        proba = model.predict_proba(df_processed)[0]
    
    label_target = label_encoders["Churn"].inverse_transform([prediksi])[0]
    
    st.subheader(f"Hasil Analisis Prediksi ({pilihan_model})")
    
    if label_target == "Yes":
        st.error("Status: Pelanggan Diprediksi CHURN (Berpotensi Berhenti Berlangganan)")
        if has_proba:
            st.write(f"Tingkat Probabilitas Churn: **{proba[1]*100:.2f}%**")
    else:
        st.success("Status: Pelanggan Diprediksi RETENTION (Tetap Berlangganan)")
        if has_proba:
            st.write(f"Tingkat Probabilitas Retensi: **{proba[0]*100:.2f}%**")