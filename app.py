import pickle
import numpy as np
from PIL import Image
import streamlit as st
import cv2
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.transform import resize
import gdown 

# Load model dan preprocessing tools

url = 'https://drive.google.com/file/d/1s8kB9izBxbFyBaPHD9_VMUW-gog1UfKD/view?usp=drive_link'
output = 'model.pkl'
gdown.download(url, output, quiet=False)

with open(output, 'rb') as f:
    artifacts = pickle.load(f)


model = artifacts['model']
scaler = artifacts['scaler']
pca = artifacts['pca']
le = artifacts['label_encoder']

st.title("Klasifikasi Fraktur Gambar Radiologi")

uploaded_file = st.file_uploader("Upload gambar radiologi...", type=["png", "jpg", "jpeg"])

# Fungsi untuk memproses gambar (resize, CLAHE, masking lingkaran)
def process_single_image(img_pil, image_size=(256, 256)):
    img_np = np.array(img_pil)
    img_resized = cv2.resize(img_np, image_size)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)

    # Masking lingkaran
    h, w = img_clahe.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(center[0], center[1], w - center[0], h - center[1])
    cv2.circle(mask, center, radius, 255, -1)
    img_masked = cv2.bitwise_and(img_clahe, img_clahe, mask=mask)

    return img_masked

# Fungsi untuk ekstraksi fitur (GLCM + HOG)
def extract_glcm_features(img_masked):
    # Pastikan ukuran sama dengan saat training
    # Jika dulu kamu training tanpa resize, kemungkinan ukuran aslinya 512x512
    img_resized = resize(img_masked, (256, 256), anti_aliasing=True)
    img_uint8 = (img_resized * 255).astype(np.uint8)

    # GLCM
    glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    glcm_features = [contrast, dissimilarity, homogeneity, energy, correlation, ASM]

    # HOG
    hog_features = hog(img_uint8, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

    combined_features = np.concatenate([glcm_features, hog_features])
    return combined_features

if uploaded_file is not None:
    # Buka dan tampilkan gambar
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Gambar di-upload", use_container_width=True)

    # Preprocess
    img_masked = process_single_image(image)

    # Ekstraksi fitur
    features = extract_glcm_features(img_masked)

    # Transformasi fitur (skalasi dan PCA)
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)

    # Prediksi
    pred = model.predict(features_pca)
    label = le.inverse_transform(pred)[0]

    st.success(f"Prediksi: **{label}**")
