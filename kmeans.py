import cv2
import numpy as np
import os
import shutil
from sklearn.cluster import KMeans

def load_images_from_directory(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte de BGR para RGB
            images.append(img)
            filenames.append(filename)
    return images, filenames

def extract_features(images):
    features = []
    for img in images:
        # Redimensiona a imagem para 128x128 para uniformidade
        img = cv2.resize(img, (128, 128))
        
        # Calcula o histograma de cores para 3 canais (RGB) e concatena
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
        hist = np.concatenate([hist_r, hist_g, hist_b])
        
        # Normaliza o histograma
        hist = hist / np.sum(hist)
        features.append(hist.flatten())
    return np.array(features)

def perform_clustering(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

def organize_images_by_cluster(directory, filenames, labels):
    # Cria os diretórios para os clusters, se não existirem
    for label in np.unique(labels):
        cluster_dir = os.path.join(directory, f'cluster_{label}')
        os.makedirs(cluster_dir, exist_ok=True)
    
    # Move as imagens para o diretório correspondente ao cluster
    for filename, label in zip(filenames, labels):
        src_path = os.path.join(directory, filename)
        dest_dir = os.path.join(directory, f'cluster_{label}')
        shutil.move(src_path, os.path.join(dest_dir, filename))

def main():
    directory = 'imagens/todas_imagens'
    images, filenames = load_images_from_directory(directory)
    features = extract_features(images)
    labels = perform_clustering(features, n_clusters=3)
    
    # Organiza as imagens em subdiretórios por cluster
    organize_images_by_cluster(directory, filenames, labels)

if __name__ == "__main__":
    main()
