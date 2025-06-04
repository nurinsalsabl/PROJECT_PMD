import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

dataset_path = 'BoneFractureYolo8'
for root, dirs, files in os.walk(dataset_path):
    print(f'📂 Folder: {root}')
    for file in files:
        print(f'   └── 📄 {file}')


def process_and_visualize_dataset(folder_path, image_size=(256, 256), n_visualize=5):

    valid_ext = ('.jpg', '.jpeg', '.png')
    all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed_data = []

    for img_path in all_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # skip corrupted or unreadable images

        img_resized = cv2.resize(img, image_size)
        img_clahe = clahe.apply(img_resized)

        h, w = img_clahe.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(center[0], center[1], w - center[0], h - center[1])
        cv2.circle(mask, center, radius, 255, -1)
        img_masked = cv2.bitwise_and(img_clahe, img_clahe, mask=mask)

        processed_data.append({
            'original': img_resized,
            'clahe': img_clahe,
            'masked': img_masked,
            'path': img_path
        })

    sample_data = random.sample(processed_data, min(n_visualize, len(processed_data)))

    fig, axs = plt.subplots(len(sample_data), 3, figsize=(14, 4 * len(sample_data)))

    for i, data in enumerate(sample_data):
        axs[i, 0].imshow(data['original'], cmap='gray')
        axs[i, 0].set_title(f'Original: {os.path.basename(data["path"])}')

        axs[i, 1].imshow(data['clahe'], cmap='gray')
        axs[i, 1].set_title('CLAHE')

        axs[i, 2].imshow(data['masked'], cmap='gray')
        axs[i, 2].set_title('Masked CLAHE')

        for j in range(3):
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

    return processed_data

processed_data_train = process_and_visualize_dataset("BoneFractureYolo8/train/images")
processed_data_test = process_and_visualize_dataset("BoneFractureYolo8/test/images")
processed_data_valid = process_and_visualize_dataset("BoneFractureYolo8/valid/images")

yaml_labels = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

def get_label_from_txt(txt_file):
    with open(txt_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            class_idx = int(first_line.split()[0])
            return yaml_labels[class_idx]
        else:
            return None  #kosong

import pandas as pd
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_hog_features_from_preprocessed(processed_data):
    features_list = []

    for data in processed_data:
        # Ambil citra (grayscale tepi)
        img = data['masked']
        img_uint8 = img.astype(np.uint8)

        # Label dari txt
        txt_file = data['path'].replace('/images/', '/labels/').replace('.png', '.txt').replace('.jpg', '.txt')
        label = get_label_from_txt(txt_file)

        ### --- GLCM ---
        glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        ASM = graycoprops(glcm, 'ASM')[0, 0]
        glcm_features = [contrast, dissimilarity, homogeneity, energy, correlation, ASM]

        ### --- HOG ---
        # Pastikan citra normalisasi/resize kalau perlu
        # Resize standar misalnya:
        # from skimage.transform import resize
        # img_uint8 = resize(img_uint8, (128, 64), anti_aliasing=True)

        # HOG: pakai orientasi, pixels_per_cell, cells_per_block sesuai dataset
        hog_features = hog(img_uint8, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

        # Gabungkan GLCM + HOG
        combined_features = np.concatenate([glcm_features, hog_features])

        # Simpan
        features_list.append({
            'filename': data['path'],
            'features': combined_features,
            'label': label
        })

    return features_list

features_train = extract_glcm_hog_features_from_preprocessed(processed_data_train)
features_test = extract_glcm_hog_features_from_preprocessed(processed_data_test)
features_valid = extract_glcm_hog_features_from_preprocessed(processed_data_valid)

df_train = pd.DataFrame(features_train)
df_test = pd.DataFrame(features_test)
df_valid = pd.DataFrame(features_valid)
# Sebelum simpan CSV

# df_train.to_csv('glcm_features_train.csv', index=False)
# df_test.to_csv('glcm_features_test.csv', index=False)
# df_valid.to_csv('glcm_features_valid.csv', index=False)

# print("Ekstraksi fitur GLCM selesai! Hasil disimpan dalam format csv")
