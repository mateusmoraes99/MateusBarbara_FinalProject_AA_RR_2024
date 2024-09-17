import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.ndimage import label
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def load_images_from_directory(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def preprocess_images(images):
    # Normaliza as imagens para valores entre 0 e 1
    processed_images = [img / 255.0 for img in images]
    return processed_images

def calculate_intensity_estimate(image, x, y, window_size=7):
    # Pega uma janela centrada em (x, y)
    half_window = window_size // 2
    window = image[max(0, x-half_window):x+half_window+1, max(0, y-half_window):y+half_window+1]
    # Calcula a média da intensidade da janela
    intensity_estimate = np.mean(window)
    return intensity_estimate

def gibbs_field(image, beta, num_iterations=5, filename=None):
    height, width = image.shape
    labels = np.random.randint(0, 3, size=(height, width))  # Inicializa com rótulos aleatórios
    intensity_estimates = np.zeros((height, width, 3))  # Para 3 categorias
    
    for i in range(num_iterations):
        print(f"Imagem {filename} - Iteração {i + 1}/{num_iterations}")
        for x in range(height):
            for y in range(width):
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                neighbor_labels = [labels[nx, ny] if 0 <= nx < height and 0 <= ny < width else -1 for nx, ny in neighbors]
                current_label = labels[x, y]
                
                # Atualiza a estimativa da intensidade com a janela centrada em (x, y)
                intensity_estimates[x, y, current_label] = calculate_intensity_estimate(image, x, y)
                
                # Calcula a probabilidade de mudança de rótulo
                probs = np.array([np.exp(-beta * intensity_estimates[x, y, l]) for l in range(3)])
                probs /= np.sum(probs)  # Normaliza as probabilidades para somarem 1
                
                # Atualiza o rótulo baseado nas probabilidades
                labels[x, y] = np.random.choice(3, p=probs)
    
    return labels

def plot_image_clusters(image, labels, filename):
    plt.imshow(labels, cmap='tab10')  # Mapa de cores para visualizar clusters
    plt.title(f'Segmented Image - {filename}')
    plt.savefig(f"clusters_{filename}")

def process_image(image_info):
    image, filename = image_info
    beta = 0.1
    labels = gibbs_field(image, beta, filename=filename)
    plot_image_clusters(image, labels, filename)

def main():
    directory = 'imagens/todas_imagens'
    images, filenames = load_images_from_directory(directory)
    processed_images = preprocess_images(images)
    
    # Usar paralelização
    num_workers = cpu_count()
    print(f"Usando {num_workers} núcleos para paralelizar.")
    
    with Pool(num_workers) as pool:
        pool.map(process_image, zip(processed_images, filenames))
        print("Paralelização completa.")

if __name__ == "__main__":
    main()
