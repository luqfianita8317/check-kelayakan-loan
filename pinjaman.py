Python 3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer

# --- Konfigurasi Halaman & UI/UX Menarik ---
st.set_page_config(
    page_title="Prediksi Status Pinjaman (Decision Tree)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Utama & Tampilan Menarik
st.title("💸 Prediksi Kelayakan Pinjaman (Decision Tree)")
st.markdown("Aplikasi interaktif ini menggunakan **Model Decision Tree** yang dilatih dengan 5 fitur terpenting untuk memprediksi *loan status*.")
st.markdown("---")

# --- 1. Metrik Kinerja yang Diberikan (Hardcoded) ---
METRICS = {
    "Accuracy": 0.9216,
    "Precision": 0.8817,
    "Recall": 0.7493,
    "F1-Score": 0.8101,
    "ROC-AUC Score": 0.9617
}

# Top 5 Fitur Paling Penting (sesuai permintaan user)
FEATURES_INPUT = ['previous_loan_defaults_on_file', 'loan_percent_income', 'loan_int_rate', 'person_income', 'person_home_ownership']
FILE_DATA = "loan_data.csv"
TARGET_COLUMN = 'loan_status' # 0 = Accepted/Dibayar, 1 = Default/Ditolak

try:
    df = pd.read_csv(FILE_DATA)
    df_clean = df.copy()

    # --- Preprocessing & Pelatihan Model ---
    
    # 1. Imputasi Bunga (Int Rate)
    mean_int_rate = df_clean['loan_int_rate'].mean()
    df_clean['loan_int_rate'].fillna(mean_int_rate, inplace=True)
    
    # 2. Encoding Fitur Biner
    df_clean['previous_loan_defaults_on_file'] = df_clean['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    
    # Pilih fitur untuk pelatihan
    X_train_data = df_clean[FEATURES_INPUT]
    y = df_clean[TARGET_COLUMN]

    # 3. One-Hot Encoding Kepemilikan Rumah
    X_processed = pd.get_dummies(X_train_data, columns=['person_home_ownership'], drop_first=True)
    X_processed = X_processed.select_dtypes(include=np.number).dropna() # Pastikan hanya numerik dan hilangkan sisa NaN
    y = y.loc[X_processed.index] # Sinkronkan target
    
    # Split data
    X_train_final, X_test_final, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Latih Model Decision Tree
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train_final, y_train)
    
    # Prediksi untuk Confusion Matrix
    y_pred = model.predict(X_test_final)
    
    # Simpan daftar kolom akhir untuk prediksi input
    model_features = X_train_final.columns.tolist()
    
    st.sidebar.success(f"Model Decision Tree berhasil dimuat dan dilatih.")

except FileNotFoundError:
    st.error(f"⚠️ Error: File '{FILE_DATA}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan saat memproses data atau melatih model: {e}")
    st.stop()


# --- 2. Implementasi Tab UI/UX ---
tab1, tab2 = st.tabs(["💡 Prediksi Loan Status", "📊 Analisis Data & Grafik"])

with tab1:
    st.header("Form Input Prediksi Pinjaman")
    st.markdown("Masukkan data 5 fitur penting pinjaman untuk mendapatkan prediksi status dari **Model Decision Tree**.")
    
    st.subheader("Top 5 Fitur Utama (Decision Tree)")
    
    # Mengatur layout input form dalam kolom
    col_input_1, col_input_2 = st.columns(2)

    with col_input_1:
        # person_income (pendapatan)
        person_income = st.number_input("Pendapatan Tahunan (person_income)", min_value=10000, max_value=300000, value=75000, step=1000)
        # loan_int_rate (bunga pinjaman)
        loan_int_rate = st.number_input("Bunga Pinjaman (%) (loan_int_rate)", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
        # previous_loan_defaults_on_file
        previous_loan_defaults_on_file = st.selectbox("Pernah Gagal Bayar Sebelumnya?", options=['No', 'Yes'])


    with col_input_2:
        # loan_percent_income (% pinjaman dari pendapatan)
        loan_percent_income = st.slider("% Pinjaman dari Pendapatan (loan_percent_income)", min_value=0.01, max_value=0.5, value=0.15, step=0.01)
        # person_home_ownership - Pilihan terbatas sesuai permintaan
        person_home_ownership = st.selectbox("Status Kepemilikan Rumah", options=['OWN', 'RENT', 'MORTGAGE'])
        
        st.info("Input ini adalah 5 fitur yang paling memengaruhi hasil prediksi Model Decision Tree.")

    # Tombol Prediksi
    if st.button("Prediksi Status Pinjaman (Decision Tree)", help="Dapatkan hasil prediksi dari Model Decision Tree."):
        
        # --- Preprocessing Data Input Baru ---
        input_data = {
            'loan_percent_income': loan_percent_income,
            'loan_int_rate': loan_int_rate,
            'person_income': person_income,
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file,
            'person_home_ownership': person_home_ownership
        }
        
        new_data_df = pd.DataFrame([input_data])
        
        # 1. Encoding Biner
        new_data_df['previous_loan_defaults_on_file'] = new_data_df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
        
        # 2. One-Hot Encoding dan Sinkronisasi Kolom
        input_processed = new_data_df.copy()
        
        # Inisialisasi semua kolom OHE yang mungkin ada dari pelatihan ke 0
        for col in [c for c in model_features if c.startswith('person_home_ownership_')]:
             input_processed[col] = 0

        # Setel kolom OHE yang sesuai dari input pengguna menjadi 1
        col_name = f'person_home_ownership_{person_home_ownership}'
        if col_name in model_features:
            input_processed[col_name] = 1

        # Drop kolom asli dan ambil hanya kolom yang digunakan model
        input_final = input_processed.drop(columns=['person_home_ownership']).select_dtypes(include=np.number)
        input_final = input_final[model_features] # Memastikan urutan kolom benar

        # Melakukan Prediksi
        prediction = model.predict(input_final)[0]
        prediction_proba = model.predict_proba(input_final)[0][1] # Probabilitas default (kelas 1)

        st.subheader("✅ Hasil Prediksi")
        
        # Asumsi: 0=DITERIMA (Non-Default), 1=DEFAULT (Ditolak)
        if prediction == 0:
            st.success(f"**Status Pinjaman Diprediksi: DITERIMA/DIBAYAR**")
            st.metric(label="Probabilitas Default (Kelas 1)", value=f"{prediction_proba*100:.2f}%")
        else:
            st.error(f"**Status Pinjaman Diprediksi: DEFAULT/DITOLAK**")
            st.metric(label="Probabilitas Default (Kelas 1)", value=f"{prediction_proba*100:.2f}%")
            
        st.markdown(f"*(**Model yang digunakan: Decision Tree Classifier**)*")


with tab2:
    st.header("Analisis Data & Evaluasi Model Decision Tree")
    st.markdown("Tab ini menampilkan kinerja dan visualisasi **Model Decision Tree**.")
    
    st.subheader("Metrik Kinerja Model Decision Tree (Nilai yang Disediakan)")
    
    # Tampilkan Metrik yang Diberikan
    col_metrics = st.columns(5)
    
    col_metrics[0].metric(label="Akurasi", value=f"{METRICS['Accuracy']:.4f}")
    col_metrics[1].metric(label="Precision", value=f"{METRICS['Precision']:.4f}")
    col_metrics[2].metric(label="Recall", value=f"{METRICS['Recall']:.4f}")
    col_metrics[3].metric(label="F1-Score", value=f"{METRICS['F1-Score']:.4f}")
    col_metrics[4].metric(label="ROC-AUC Score", value=f"{METRICS['ROC-AUC Score']:.4f}")
    
    st.markdown(f"*(Metrik di atas berasal dari hasil evaluasi **Model Decision Tree** sebelumnya)*")
...     st.markdown("---")
...     
...     st.subheader("Visualisasi Utama Model Decision Tree")
... 
...     # Visualisasi 1: Confusion Matrix
...     st.markdown("#### 1. Confusion Matrix (Decision Tree)")
...     st.caption("Membandingkan label sebenarnya (True Label) dengan prediksi model (Predicted Label).")
...     cm = confusion_matrix(y_test, y_pred)
...     
...     fig1, ax1 = plt.subplots(figsize=(6, 6))
...     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
...                 xticklabels=['Dibayar (0)', 'Default (1)'], 
...                 yticklabels=['Dibayar (0)', 'Default (1)'], ax=ax1)
...     ax1.set_title('Confusion Matrix (Decision Tree)')
...     ax1.set_ylabel('True Label')
...     ax1.set_xlabel('Predicted Label')
...     st.pyplot(fig1)
...     plt.close(fig1) # Pembersihan plot
... 
...     # Visualisasi 2: Feature Importances Heatmap
...     st.markdown("#### 2. Feature Importances Heatmap (Decision Tree)")
...     st.caption("Menunjukkan seberapa besar kontribusi setiap fitur dalam proses pengambilan keputusan model.")
...     
...     # Dapatkan feature importance dari model yang dilatih
...     feature_importance = model.feature_importances_
...     
...     # Buat Series dan normalize/frame
...     importance_series = pd.Series(feature_importance, index=X_train_final.columns).sort_values(ascending=False)
...     importance_matrix = importance_series.to_frame(name='Importance').T
...     
...     fig2, ax2 = plt.subplots(figsize=(12, 3))
...     sns.heatmap(importance_matrix, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Kepentingan Fitur'}, ax=ax2)
...     ax2.set_title('Feature Importances Heatmap (Decision Tree)')
...     ax2.set_yticks([]) 
...     st.pyplot(fig2)
