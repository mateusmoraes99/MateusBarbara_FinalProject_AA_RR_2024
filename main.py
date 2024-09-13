import os
import cv2
import numpy as np
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import color
from matplotlib import pyplot as plt

def resize_image(image, size=(128, 128)):
    return cv2.resize(image, size)

def extract_dominant_colors(image, k=3):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)

    # Perform K-Means clustering to find dominant colors
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    return colors.flatten()  # Flatten to 1D array

def extract_color_histogram(image, bins=(16, 16, 16)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_hog_features(image):
    gray_image = color.rgb2gray(image)
    features, hog_image = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True, feature_vector=True)
    return features  # Return features as 1D array

def cluster_images(image_features, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(image_features)
    return kmeans.labels_, kmeans.cluster_centers_

def plot_cluster_distribution(features_array, labels, output_path):
    # Apply PCA to reduce the dimensionality to 2D for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_array)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Distribuição das Imagens nos Clusters')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.tight_layout()  # Adjust layout
    plt.savefig(output_path)
    plt.close()

def create_cluster_folders(output_folder, num_clusters):
    for i in range(num_clusters):
        cluster_folder = os.path.join(output_folder, f'cluster_{i}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

def copy_images_to_clusters(image_paths, labels, output_folder):
    num_clusters = len(set(labels))
    create_cluster_folders(output_folder, num_clusters)

    for image_path, label in zip(image_paths, labels):
        cluster_folder = os.path.join(output_folder, f'cluster_{label}')
        shutil.copy(image_path, cluster_folder)

def load_images_and_extract_features(image_folder, output_folder):
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.jpeg'))]
    
    if len(image_paths) == 0:
        raise ValueError("Nenhuma imagem foi encontrada na pasta especificada.")
    
    features_list = []
    for image_path in image_paths:
        print(f"Carregando imagem: {image_path}")
        image = cv2.imread(image_path)
        if image is not None:
            image = resize_image(image)
            dominant_colors = extract_dominant_colors(image)
            hist = extract_color_histogram(image)
            hog_features = extract_hog_features(image)

            # Combine features into a single vector
            combined_features = np.concatenate((dominant_colors, hist, hog_features))
            features_list.append(combined_features)
        else:
            print(f"Erro ao carregar imagem: {image_path}")

    # Convert features list to array
    features_array = np.array(features_list)

    # Perform clustering
    labels, cluster_centers = cluster_images(features_array, num_clusters=3)

    # Plot cluster distribution
    plot_cluster_distribution(features_array, labels, 'cluster_distribution.png')

    # Copy images to cluster folders
    copy_images_to_clusters(image_paths, labels, output_folder)

    print("Clusterização concluída, gráfico salvo como 'cluster_distribution.png' e imagens copiadas para pastas dos clusters.")

# Exemplo: Pasta onde suas imagens estão e onde salvar as saídas
image_folder = 'imagens/todas_imagens'
output_folder = 'imagens/clusterização_resultados'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

load_images_and_extract_features(image_folder, output_folder)
