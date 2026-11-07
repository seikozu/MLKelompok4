import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(
    page_title="Prediksi Harga Rumah California",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------
# LOAD SEMUA MODEL & METADATA
# ----------------------------------------------------
@st.cache_resource
def load_all_models_and_metadata():
    models = {}
    
    # 1. Path AdaBoost
    adaboost_path = os.path.join(BASE_DIR, "saved_models", "adaboost_model.joblib")
    if os.path.exists(adaboost_path):
        try:
            models["AdaBoost (Adaptive Boosting)"] = joblib.load(adaboost_path)
        except Exception as e:
            st.error(f"Gagal memuat AdaBoost: {e}")
            
    # 2. Path Gradient Boosting
    gb_path = os.path.join(BASE_DIR, "saved_models", "best_model.joblib")
    if os.path.exists(gb_path):
        try:
            models["Gradient Boosting"] = joblib.load(gb_path)
        except Exception as e:
            st.error(f"Gagal memuat Gradient Boosting: {e}")
            
    # Load Metadata JSON
    meta_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
    metadata = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            pass
            
    return models, metadata

@st.cache_data
def load_ref_data():
    housing_path = os.path.join(BASE_DIR, "housing.csv")
    if os.path.exists(housing_path):
        return pd.read_csv(housing_path)
    return None

available_models, metadata = load_all_models_and_metadata()
df_ref = load_ref_data()

# Metrik Acuan Manual (jika metadata belum lengkap)
DEFAULT_METRICS = {
    "Gradient Boosting": {"MAE": 31692.86, "RMSE": 47703.60, "MAPE (%)": 18.01, "R2": 0.8263},
    "AdaBoost (Adaptive Boosting)": {"MAE": 40747.71, "RMSE": 56530.53, "MAPE (%)": 24.43, "R2": 0.7561}
}

# ----------------------------------------------------
# 1. BILAH SISI (SIDEBAR): PILIH MODEL & METRIKS
# ----------------------------------------------------
with st.sidebar:
    st.header("Pengaturan & Metrik Model")
    
    # Pilihan Model
    selected_model_name = st.selectbox(
        "Pilih Algoritma Model Regresi:",
        options=["Gradient Boosting", "AdaBoost (Adaptive Boosting)"],
        help="Pilih model yang akan digunakan untuk memprediksi."
    )
    
    current_model = available_models.get(selected_model_name, None)
    
    st.write("---")
    st.subheader(f"Model Aktif: {selected_model_name}")
    st.markdown("**Ringkasan Metrik Evaluasi:**")
    
    # Ambil metrik sesuai pilihan model di sidebar
    metrics = None
    if metadata and "evaluation_metrics" in metadata:
        # Cek kunci di metadata (bisa "Gradient Boosting" / "AdaBoost")
        key_name = "Gradient Boosting" if "Gradient" in selected_model_name else "AdaBoost"
        metrics = metadata["evaluation_metrics"].get(key_name, None)
        
    if not metrics:
        metrics = DEFAULT_METRICS.get(selected_model_name, {})
        
    st.metric(label="MAE", value=f"{metrics.get('MAE', 0):,.2f}")
    st.metric(label="RMSE", value=f"{metrics.get('RMSE', 0):,.2f}")
    st.metric(label="MAPE", value=f"{metrics.get('MAPE (%)', 0):.2f}%")
    st.metric(label="R² Score", value=f"{metrics.get('R2', 0):.4f}")

# ----------------------------------------------------
# 2. AREA UTAMA: JUDUL & DESKRIPSI
# ----------------------------------------------------
st.title("Prediksi Harga Rumah California")
st.write(f"Aplikasi simulasi prediksi harga rumah menggunakan algoritma **{selected_model_name}**.")

# ----------------------------------------------------
# 3. FORMULIR INPUT (9 FITUR)
# ----------------------------------------------------
st.header("Formulir Input Fitur Rumah")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("longitude", value=float(df_ref['longitude'].median()) if df_ref is not None else -118.49)
    latitude = st.number_input("latitude", value=float(df_ref['latitude'].median()) if df_ref is not None else 34.26)
    housing_median_age = st.number_input("housing_median_age", value=float(df_ref['housing_median_age'].median()) if df_ref is not None else 29.0)
    total_rooms = st.number_input("total_rooms", value=float(df_ref['total_rooms'].median()) if df_ref is not None else 2127.0)
    total_bedrooms = st.number_input("total_bedrooms", value=float(df_ref['total_bedrooms'].median()) if df_ref is not None else 435.0)

with col2:
    population = st.number_input("population", value=float(df_ref['population'].median()) if df_ref is not None else 1166.0)
    households = st.number_input("households", value=float(df_ref['households'].median()) if df_ref is not None else 409.0)
    median_income = st.number_input("median_income", value=float(df_ref['median_income'].median()) if df_ref is not None else 3.53)
    
    ocean_options = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    if df_ref is not None and 'ocean_proximity' in df_ref.columns:
        ocean_options = list(df_ref['ocean_proximity'].dropna().unique())
    ocean_proximity = st.selectbox("ocean_proximity", options=ocean_options)

# ----------------------------------------------------
# 4. TOMBOL PREDIKSI & 5. PANEL HASIL
# ----------------------------------------------------
st.write("---")
if st.button("Jalankan Prediksi Harga", type="primary"):
    if current_model is not None:
        try:
            feature_cols = [
                'longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'
            ]
            
            input_df = pd.DataFrame([{
                'longitude': longitude,
                'latitude': latitude,
                'housing_median_age': housing_median_age,
                'total_rooms': total_rooms,
                'total_bedrooms': total_bedrooms,
                'population': population,
                'households': households,
                'median_income': median_income,
                'ocean_proximity': ocean_proximity
            }])[feature_cols]
            
            prediction = current_model.predict(input_df)[0]
            mae_val = metrics.get('MAE', 31693.0)
            
            # Panel Hasil Dinamis
            st.success(f"Prediksi Berhasil Dijalankan Menggunakan Model: **{selected_model_name}**!")
            st.metric(label="Estimasi Prediksi Harga Rumah", value=f"${prediction:,.2f}")
            st.info(f"💡 **Catatan:** Rata-rata tingkat kesalahan (margin of error) dari model **{selected_model_name}** adalah sekitar **MAE (± ${mae_val:,.2f})** berdasarkan data uji.")
            
        except Exception as e:
            st.error(f"Gagal memproses prediksi: {e}")
    else:
        st.error(f"Model **{selected_model_name}** tidak tersedia atau gagal dimuat.")