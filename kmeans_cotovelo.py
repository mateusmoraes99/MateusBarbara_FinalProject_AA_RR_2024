import os
import cv2
import numpy as np
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import color
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

def resize_image(image, size=(128, 128)):
    return cv2.resize(image, size)

def extract_dominant_colors(image, k=3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    return colors.flatten()

def extract_color_histogram(image, bins=(16, 16, 16)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_hog_features(image):
    gray_image = color.rgb2gray(image)
    features = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, feature_vector=True)
    return features

def determine_optimal_clusters(features_array, max_clusters=10):
    sse = []
    silhouette_scores = []
    k_range = range(1, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(features_array)
        sse.append(kmeans.inertia_)
        
        if k > 1:
            labels = kmeans.labels_
            score = silhouette_score(features_array, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)  # Silhouette score is not defined for k=1
    
    # Plot SSE for the Elbow Method
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse, marker='o')
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('SSE')
    
    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_range[1:], silhouette_scores[1:], marker='o')  # Skip k=1 for silhouette
    plt.title('Score de Silhueta')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Score de Silhueta')
    
    plt.tight_layout()
    plt.savefig('cluster_evaluation.png')
    plt.show()
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k

def cluster_images(image_features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(image_features)
    return kmeans.labels_, kmeans.cluster_centers_

def plot_cluster_distribution(features_array, labels, output_path):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_array)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Distribuição das Imagens nos Clusters')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.tight_layout()
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
            combined_features = np.concatenate((dominant_colors, hist, hog_features))
            features_list.append(combined_features)
        else:
            print(f"Erro ao carregar imagem: {image_path}")

    features_array = np.array(features_list)

    # Determine the optimal number of clusters
    optimal_clusters = determine_optimal_clusters(features_array, max_clusters=10)
    print(f"Número ideal de clusters: {optimal_clusters}")

    # Perform clustering
    labels, cluster_centers = cluster_images(features_array, num_clusters=optimal_clusters)

    # Plot cluster distribution
    plot_cluster_distribution(features_array, labels, 'cluster_distribution_cotovelo.png')

    # Copy images to cluster folders
    copy_images_to_clusters(image_paths, labels, output_folder)

    print("Clusterização concluída, gráfico salvo como 'cluster_distribution.png' e imagens copiadas para pastas dos clusters.")

# Exemplo: Pasta onde suas imagens estão e onde salvar as saídas
image_folder = 'imagens/todas_imagens'
output_folder = 'clusterizacao_resultados'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

load_images_and_extract_features(image_folder, output_folder)
