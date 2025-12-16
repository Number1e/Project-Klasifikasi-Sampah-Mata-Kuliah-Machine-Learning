import os

# ==========================================
# 1. KONFIGURASI ENVIRONMENT (WAJIB DI ATAS)
# ==========================================
# Ini memperbaiki error "Layer dense expects 1 input..." pada MobileNet
# Harus dijalankan SEBELUM import tensorflow/streamlit
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# 2. IMPORT MODUL SENDIRI
# ==========================================
# Pastikan src/predict.py juga sudah diperbaiki sesuai instruksi sebelumnya
from src.predict import load_inference_model, preprocess_image, CLASS_NAMES

# ==========================================
# 3. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Trash Classification AI (Multi-Item)",
    page_icon="♻️",
    layout="wide"
)

# ==========================================
# 4. SIDEBAR: PENGATURAN MODEL
# ==========================================
st.sidebar.title("⚙️ Konfigurasi Model")
st.sidebar.info("Pilih arsitektur model yang ingin digunakan.")

# Pilihan Model
model_type = st.sidebar.selectbox(
    "Pilih Arsitektur:",
    ('CNN Scratch', 'Transfer Learning (MobileNet)', 'Vision Transformer (ViT)', 'SVM')
)

# Mapping pilihan UI ke nama file & tipe backend
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
    model_path = 'models/SVM_Classic/'
    backend_type = 'svm'

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title("♻️ Klasifikasi Sampah Cerdas (Multi-Upload)")
st.markdown(f"""
Mode Arsitektur Aktif: **{model_type}**
Upload satu atau banyak gambar sampah (Plastik, Kaca, Kertas, dll) sekaligus.
""")

st.write("---")

# ==========================================
# 6. UPLOAD SECTION (MULTI-FILE)
# ==========================================
uploaded_files = st.file_uploader(
    "📸 Pilih gambar (bisa blok banyak file sekaligus)...", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    n_files = len(uploaded_files)
    st.info(f"📁 {n_files} gambar berhasil dimuat. Klik tombol di bawah untuk memproses.")
    
    # Tombol Eksekusi
    if st.button(f'Mulai Klasifikasi ({n_files} Gambar)', type='primary'):
        
        # --- LOAD MODEL (Hanya 1x agar cepat) ---
        model_loaded = False
        model = None
        
        with st.spinner(f'Sedang memuat model {model_type}...'):
            try:
                model = load_inference_model(model_path, backend_type)
                model_loaded = True
            except Exception as e:
                st.error(f"FATAL: Gagal memuat model. Pesan Error: {e}")
                st.stop()

        if model_loaded:
            st.write("---")
            progress_bar = st.progress(0)
            
            # --- LOOPING GAMBAR ---
            for i, uploaded_file in enumerate(uploaded_files):
                
                # Layout: Kiri (Gambar), Kanan (Hasil)
                col_img, col_res = st.columns([1, 2])
                
                with col_img:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption=f"File {i+1}: {uploaded_file.name}", use_container_width=True)

                with col_res:
                    temp_path = f"temp_img_{i}.jpg"
                    try:
                        # A. Simpan file sementara
                        image.save(temp_path)
                        
                        # B. Tentukan Ukuran (FIX UNTUK VIT vs MOBILENET)
                        if backend_type == 'vit':
                            current_target_size = (128, 128) # ViT butuh 128
                        else:
                            current_target_size = (224, 224) # MobileNet, CNN butuh 224
                            # SVM ukurannya dihandle internal pipeline (biasanya resize sendiri)
                        
                        # C. Preprocess
                        raw_img_array = preprocess_image(temp_path, current_target_size)
                        
                        # D. Prediksi
                        predicted_class = "Unknown"
                        confidence = 0.0
                        
                        # --- JALUR SVM ---
                        if backend_type == 'svm':
                            idx, conf, _ = model.predict(raw_img_array)
                            predicted_class = CLASS_NAMES[int(idx)]
                            confidence = conf 
                            
                            st.subheader(f"🏷️ Hasil: {predicted_class}")
                            st.info("ℹ️ Menggunakan Ekstraksi Fitur SVM")

                        # --- JALUR DEEP LEARNING ---
                        else:
                            # Normalisasi & Expand Dims
                            img_tensor = raw_img_array.astype('float32') / 255.0
                            img_tensor = np.expand_dims(img_tensor, axis=0)
                            
                            predictions = model.predict(img_tensor)
                            idx = np.argmax(predictions[0])
                            confidence = np.max(predictions[0])
                            predicted_class = CLASS_NAMES[idx]
                            
                            st.subheader(f"🏷️ Hasil: {predicted_class}")
                            st.caption(f"Confidence: {confidence*100:.1f}%")
                            
                            # Chart
                            st.bar_chart({
                                "Kelas": CLASS_NAMES, 
                                "Probabilitas": predictions[0]
                            }, x="Kelas", y="Probabilitas", height=180)

                    except Exception as e:
                        st.error(f"Gagal memproses {uploaded_file.name}: {e}")
                    
                    finally:
                        # E. Bersihkan file sementara
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                st.divider() # Garis pemisah antar item
                progress_bar.progress((i + 1) / n_files)
            
            st.success("✅ Semua proses selesai!")

else:
    st.info("👋 Silakan upload gambar sampah di panel atas untuk memulai.")

# Footer
st.markdown("---")
st.caption("Project Klasifikasi Sampah - ViT vs CNN vs MobileNet vs SVM")