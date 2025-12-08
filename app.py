import streamlit as st
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# PERBAIKAN IMPORT DI SINI
# ==========================================
# Kita ganti IMG_SIZE menjadi IMG_SIZE_DL
from src.predict import load_inference_model, preprocess_image, SVMPipeline, CLASS_NAMES, IMG_SIZE_DL

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Trash Classification AI",
    page_icon="♻️",
    layout="wide"
)

# ==========================================
# SIDEBAR: PENGATURAN MODEL
# ==========================================
st.sidebar.title("⚙️ Konfigurasi Model")
st.sidebar.info("Pilih arsitektur model yang ingin digunakan untuk prediksi.")

# Pilihan Model
model_type = st.sidebar.selectbox(
    "Pilih Arsitektur:",
    ('CNN Scratch', 'Transfer Learning (MobileNet)', 'Vision Transformer (ViT)', 'SVM')
)

# Mapping pilihan UI ke nama file/tipe internal
if model_type == 'CNN Scratch':
    model_path = 'models/cnn_scratch_sampah.keras'
    backend_type = 'cnn'
elif model_type == 'Transfer Learning (MobileNet)':
    model_path = 'models/mobilenet_sampah_best.h5'
    backend_type = 'mobilenet'
elif model_type == 'Vision Transformer (ViT)':
    model_path = 'models/vit_mini_waste_classifier.keras'
    backend_type = 'vit'
elif model_type == 'SVM':
    model_path = 'models/svm_best/'
    backend_type = 'svm'

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("♻️ Klasifikasi Sampah Cerdas")
st.markdown("""
Aplikasi ini menggunakan Deep Learning untuk mendeteksi jenis sampah.
Upload gambar sampah (Plastik, Kaca, Kertas, dll) untuk melihat hasilnya.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Gambar")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Tampilkan gambar
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang diupload', use_container_width=True)

with col2:
    st.subheader("🔍 Hasil Analisis")
    
    if uploaded_file is not None:
        # Tombol Prediksi
        if st.button('Mulai Klasifikasi', type='primary'):
            
            with st.spinner(f'Sedang memproses menggunakan {model_type}...'):
                try:
                    # 1. Simpan file sementara
                    temp_path = "temp_img.jpg"
                    image.save(temp_path)
                    
                    # 2. Preprocess
                    # Gunakan IMG_SIZE_DL sebagai ukuran standar tampilan/load awal
                    # (Logic internal SVM nanti akan me-resize ulang sendiri ke 200x200)
                    raw_img_array = preprocess_image(temp_path, IMG_SIZE_DL)
                    
                    # 3. Load Model
                    model = load_inference_model(model_path, backend_type)
                    
                    # 4. Inferensi
                    predicted_class = "Unknown"
                    confidence = 0.0
                    
                    if backend_type == 'svm':
                        # Logika SVM (Menerima raw array, nanti diekstrak fiturnya didalam class SVMPipeline)
                        # Perhatikan: SVMPipeline.predict sekarang return 3 values (idx, conf, detail)
                        idx, conf, _ = model.predict(raw_img_array)
                        
                        predicted_class = CLASS_NAMES[int(idx)]
                        confidence = conf # Bisa 1.0 atau probability jika SVM support
                        
                        st.success("Selesai!")
                        st.metric(label="Prediksi Kelas", value=predicted_class)
                        st.info("ℹ️ Model SVM menggunakan ekstraksi fitur manual (Warna, Tekstur GLCM, Bentuk HuMoments).")
                        
                    else:
                        # Logika Deep Learning
                        img_tensor = raw_img_array.astype('float32') / 255.0
                        img_tensor = np.expand_dims(img_tensor, axis=0)
                        
                        predictions = model.predict(img_tensor)
                        idx = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        predicted_class = CLASS_NAMES[idx]
                        
                        st.success("Selesai!")
                        
                        # Tampilkan Metric Utama
                        st.metric(label="Prediksi Kelas", value=predicted_class, delta=f"{confidence*100:.1f}% Akurat")
                        
                        # Tampilkan Chart Probabilitas
                        st.write("---")
                        st.caption("Distribusi Probabilitas:")
                        st.bar_chart({
                            "Kelas": CLASS_NAMES, 
                            "Probabilitas": predictions[0]
                        }, x="Kelas", y="Probabilitas")
                    
                    # Hapus file sementara
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
                    st.error("Pastikan path model benar, file model diupload, dan class 'Patches' (untuk ViT) sudah terdefinisi.")
    else:
        st.info("Silakan upload gambar di panel sebelah kiri untuk memulai.")

# Footer
st.markdown("---")
st.caption("Dibuat sebagai Project Klasifikasi Sampah - ViT vs CNN vs Transfer Learning vs SVM")
