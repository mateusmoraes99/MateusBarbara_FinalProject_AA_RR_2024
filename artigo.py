import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.ndimage import label
import matplotlib.pyplot as plt

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images):
    # Normaliza as imagens para valores entre 0 e 1
    processed_images = [img / 255.0 for img in images]
    return processed_images

def calculate_intensity_estimate(image, x, y, window_size=7):
    half_window = window_size // 2
    padded_image = np.pad(image, pad_width=half_window, mode='constant', constant_values=0)
    window = padded_image[x:x+window_size, y:y+window_size]
    return np.mean(window)

def gibbs_field(image, beta, num_iterations=10):
    height, width = image.shape
    labels = np.random.randint(0, 3, size=(height, width))  # Inicializa com rótulos aleatórios
    intensity_estimates = np.zeros((height, width, 3))  # Para 3 categorias
    
    for i in range(num_iterations):
        print("Iteração: ", i)
        for x in range(height):
            for y in range(width):
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                neighbor_labels = [labels[nx, ny] if 0 <= nx < height and 0 <= ny < width else -1 for nx, ny in neighbors]
                current_label = labels[x, y]
                
                # Atualiza a estimativa da intensidade
                window_size = 7
                intensity_estimates[x, y, current_label] = calculate_intensity_estimate(image, x, y, window_size)
                
                # Calcula a probabilidade de mudança de rótulo
                probs = np.zeros(3)  # Para 3 categorias
                for l in range(3):
                    # Calcula a energia para o rótulo atual
                    intensity = calculate_intensity_estimate(image, x, y, window_size)
                    energy = -beta * (intensity - intensity_estimates[x, y, l])
                    probs[l] = np.exp(energy)
                
                probs /= np.sum(probs)  # Normaliza para obter probabilidades válidas
                
                # Atualiza o rótulo baseado nas probabilidades
                labels[x, y] = np.random.choice(3, p=probs)
    
    return labels


def plot_image_clusters(image, labels, output_filename="clusters_artigo.png"):
    plt.imshow(labels, cmap='tab10')  # Mapa de cores para visualizar clusters
    plt.title('Segmented Image')
    plt.savefig(output_filename)
    plt.close()  # Fecha a figura para liberar memória

def main():
    directory = 'imagens/todas_imagens'
    images = load_images_from_directory(directory)
    processed_images = preprocess_images(images)
    
    for i, img in enumerate(processed_images):
        beta = 0.1
        labels = gibbs_field(img, beta)
        output_filename = f"clusters_image_{i}.png"
        plot_image_clusters(img, labels, output_filename)

if __name__ == "__main__":
    main()
