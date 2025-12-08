import os
import argparse
import numpy as np
import tensorflow as tf
import joblib
import cv2 # Library OpenCV
from PIL import Image
from skimage.feature import graycomatrix, graycoprops # Library Texture

# ==========================================
# KONFIGURASI
# ==========================================
CLASS_NAMES = ['Kaca', 'Kardus', 'Plastik'] # Sesuaikan dengan LABEL_KELAS training Anda
IMG_SIZE_DL = (224, 224) # Ukuran standar untuk Deep Learning (ViT/CNN)
IMG_SIZE_SVM = (200, 200) # Ukuran KHUSUS SVM sesuai kode training Anda

# ==========================================
# 1. FUNGSI EKSTRAKSI FITUR (COPY DARI TRAINING)
# ==========================================
def ekstrak_fitur_warna(gambar_hsv):
    hist_hue = cv2.calcHist([gambar_hsv], [0], None, [16], [0, 180])
    cv2.normalize(hist_hue, hist_hue)
    hist_sat = cv2.calcHist([gambar_hsv], [1], None, [16], [0, 256])
    cv2.normalize(hist_sat, hist_sat)
    return np.concatenate([hist_hue, hist_sat]).flatten()

def ekstrak_fitur_tekstur(gambar_gray):
    gambar_gray = (gambar_gray).astype(np.uint8)
    # Menggunakan parameter yang SAMA PERSIS dengan training
    glcm = graycomatrix(gambar_gray, [5], [0], levels=256, symmetric=True, normed=True)
    fitur_tekstur = np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ])
    return fitur_tekstur

def ekstrak_fitur_bentuk(gambar_gray):
    _, thresh = cv2.threshold(gambar_gray, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments)
    # Log transform untuk handle scale
    # Tambahkan epsilon kecil untuk menghindari log(0)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments.flatten()

def proses_fitur_svm(image_rgb):
    """
    Pipeline pengolah gambar mentah menjadi 44 fitur SVM
    """
    # 1. Resize sesuai training (200, 200)
    img_resized = cv2.resize(image_rgb, IMG_SIZE_SVM)
    
    # 2. Konversi Warna (Training pakai cv2.imread yg BGR, jadi kita ubah RGB->BGR dulu)
    # Agar histogram warnanya cocok dengan data latih
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    
    # 3. Konversi ruang warna untuk ekstraksi
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 4. Ekstraksi
    fitur_warna = ekstrak_fitur_warna(img_hsv)      # 32 fitur
    fitur_tekstur = ekstrak_fitur_tekstur(img_gray) # 5 fitur
    fitur_bentuk = ekstrak_fitur_bentuk(img_gray)   # 7 fitur

    # 5. Gabung (Total 44 fitur)
    vektor_fitur_global = np.hstack([fitur_warna, fitur_tekstur, fitur_bentuk])
    return vektor_fitur_global

# ==========================================
# 2. CLASS WRAPPER UNTUK SVM
# ==========================================
class SVMPipeline:
    def __init__(self, model_folder):
        print(f"Loading SVM components from {model_folder}...")
        try:
            self.imputer = joblib.load(os.path.join(model_folder, 'imputer.pkl'))
            self.scaler = joblib.load(os.path.join(model_folder, 'scaler.pkl'))
            self.model = joblib.load(os.path.join(model_folder, 'svm_sampah_model.pkl'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Gagal memuat komponen SVM. Pastikan folder berisi imputer.pkl, scaler.pkl, dan svm_sampah_model.pkl. Error: {e}")

    def predict(self, image_rgb):
        # 1. Ekstraksi Fitur (Mengubah gambar jadi 44 angka)
        features_vector = proses_fitur_svm(image_rgb)
        
        # 2. Reshape jadi (1, 44) karena Scikit-learn butuh batch dimension
        features_batch = features_vector.reshape(1, -1)
        
        # 3. Pipeline: Imputer -> Scaler -> Model
        data_imputed = self.imputer.transform(features_batch)
        data_scaled = self.scaler.transform(data_imputed)
        
        # 4. Prediksi
        prediction_idx = self.model.predict(data_scaled)[0]
        
        # Coba ambil probabilitas jika model support probability=True
        try:
            probs = self.model.predict_proba(data_scaled)[0]
            confidence = np.max(probs)
        except:
            confidence = 1.0 # Default jika tidak ada probabilitas
            
        return prediction_idx, confidence, None # None untuk probs detail (opsional)

# ==========================================
# 3. TAMBAHAN KHUSUS ViT (CUSTOM LAYERS)
# ==========================================
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size=6, **kwargs): 
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(
            patches,
            (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=256, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection.units})
        return config

# ==========================================
# 4. FUNGSI UTAMA LOAD MODEL
# ==========================================
def load_inference_model(model_path, model_type):
    if model_type == 'svm':
        return SVMPipeline(model_path)
    
    elif model_type in ['cnn', 'mobilenet']:
        print(f"Loading Keras model from {model_path}...")
        return tf.keras.models.load_model(model_path)
        
    elif model_type == 'vit':
        print(f"Loading ViT model with Custom Layers from {model_path}...")
        return tf.keras.models.load_model(model_path, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
    
    else:
        raise ValueError("Tipe model tidak dikenali.")

# ==========================================
# 5. FUNGSI PREPROCESSING UMUM
# ==========================================
def preprocess_image(image_path, target_size):
    """
    Load image dan return dalam format Array RGB
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {image_path}")
    img = Image.open(image_path).convert('RGB')
    # Resize awal (akan diresize ulang di dalam pipeline masing-masing jika perlu)
    img = img.resize(target_size) 
    img_array = np.array(img)
    return img_array

# ==========================================
# 6. MAIN INFERENCE LOGIC
# ==========================================
def predict(args):
    # Load Image (Raw Array RGB)
    raw_img_array = preprocess_image(args.image, IMG_SIZE_DL)
    
    # Load Model
    model = load_inference_model(args.model_path, args.model_type)
    
    predicted_class = "Unknown"
    confidence = 0.0
    
    if args.model_type == 'svm':
        # SVM punya pipeline khusus (input array RGB -> ekstraksi -> prediksi)
        idx, conf, _ = model.predict(raw_img_array)
        predicted_class = CLASS_NAMES[int(idx)]
        confidence = conf
        print(f"\n--- HASIL PREDIKSI (SVM) ---")
        
    else: 
        # Deep Learning (Input Tensor / 255.0)
        img_tensor = raw_img_array.astype('float32') / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        predictions = model.predict(img_tensor)
        idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = CLASS_NAMES[idx]
        print(f"\n--- HASIL PREDIKSI ({args.model_type.upper()}) ---")

    print(f"File       : {args.image}")
    print(f"Prediksi   : {predicted_class}")
    print(f"Confidence : {confidence*100:.2f}%")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['cnn', 'mobilenet', 'vit', 'svm'])
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    predict(args)
