#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace Eigen;
namespace fs = std::filesystem;

// Função para extrair histogramas de cores
vector<double> extract_color_histogram(const Mat& image, int bins = 32) {  
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_BGR2HSV);

    int histSize[] = { bins, bins, bins };
    float h_ranges[] = { 0, 180 };  // Hues em OpenCV vão de 0 a 180
    float s_ranges[] = { 0, 256 };
    float v_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges, v_ranges };
    int channels[] = { 0, 1, 2 };

    Mat hist;
    calcHist(&hsv_image, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return vector<double>(hist.begin<float>(), hist.end<float>());
}

// Função para extrair cores dominantes
vector<double> extract_dominant_colors(const Mat& image, int k = 5) {  // Aumente o número de clusters para encontrar mais cores dominantes
    Mat image_rgb;
    cvtColor(image, image_rgb, COLOR_BGR2RGB);

    // Redimensiona a imagem para uma matriz de pixels
    Mat pixels = image_rgb.reshape(1, image_rgb.total());
    pixels.convertTo(pixels, CV_32F);

    // Aplicando K-means para encontrar as cores dominantes
    Mat labels, centers;
    kmeans(pixels, k, labels, TermCriteria(TermCriteria::EPS 
    + TermCriteria::COUNT, 100, 0.2), 3, KMEANS_PP_CENTERS, centers);

    // Extraindo as cores dominantes
    vector<double> dominant_colors;
    for (int i = 0; i < k; ++i) {
        Vec3f color = centers.at<Vec3f>(i, 0);
        dominant_colors.push_back(color[0]); // R
        dominant_colors.push_back(color[1]); // G
        dominant_colors.push_back(color[2]); // B
    }

    return dominant_colors;
}

vector<double> extract_hog_features(const Mat& image) {
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    Size winSize(64, 128);  // Tamanho da janela de HOG
    Size blockSize(16, 16); // Tamanho dos blocos
    Size blockStride(8, 8); // Deslocamento dos blocos
    Size cellSize(8, 8);    // Tamanho das células

    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9, 1, -1, HOGDescriptor::L2Hys, 0.2);

    vector<float> descriptors;
    hog.compute(gray_image, descriptors, Size(8, 8), Size(0, 0));

    return vector<double>(descriptors.begin(), descriptors.end());
}

// Função para converter de Eigen::MatrixXd para cv::Mat
Mat eigen_to_cv(const MatrixXd& eigen_matrix) {
    Mat cv_matrix(eigen_matrix.rows(), eigen_matrix.cols(), CV_64F);
    for (int i = 0; i < eigen_matrix.rows(); ++i) {
        for (int j = 0; j < eigen_matrix.cols(); ++j) {
            cv_matrix.at<double>(i, j) = eigen_matrix(i, j);
        }
    }
    return cv_matrix;
}

// Função para normalizar características
Mat normalize_features(const Mat& features) {
    Mat normalized_features;
    normalize(features, normalized_features, 0, 1, NORM_MINMAX);
    return normalized_features;
}

// Função para clusterizar imagens usando K-Means
pair<vector<int>, Mat> cluster_images(const Mat& features, int num_clusters = 3) {
    Mat features_32f;
    features.convertTo(features_32f, CV_32F);

    // Normalizar características
    Mat normalized_features = normalize_features(features_32f);

    Mat labels, centers;
    kmeans(normalized_features, num_clusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2), 3, KMEANS_PP_CENTERS, centers);

    vector<int> labels_vec;
    labels_vec.assign((int*)labels.datastart, (int*)labels.dataend);

    return { labels_vec, centers };
}

// Função para redimensionar imagem
Mat resize_image(const Mat& image, Size new_size) {
    Mat resized_image;
    resize(image, resized_image, new_size);
    return resized_image;
}

// Modifique a função de carregamento e extração de características
void load_images_and_extract_features(const string& dir_path, const vector<string>& extensions, const string& output_folder) {
    vector<Mat> images;
    vector<vector<double>> all_features;
    vector<string> image_filenames;

    Size fixed_size(128, 128); // Tamanho fixo para redimensionamento

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        string ext = entry.path().extension().string();
        if (find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
            cout << "Carregando imagem: " << entry.path().string() << endl;
            Mat image = imread(entry.path().string());

            if (image.empty()) {
                cerr << "Falha ao carregar imagem: " << entry.path().string() << endl;
                continue;
            }

            // Redimensionar imagem para o tamanho fixo
            Mat resized_image = resize_image(image, fixed_size);

            images.push_back(resized_image);
            image_filenames.push_back(entry.path().string());

            // Extrair características
            vector<double> color_histogram = extract_color_histogram(resized_image);
            vector<double> dominant_colors = extract_dominant_colors(resized_image);
            vector<double> hog_features = extract_hog_features(resized_image);

            // Concatenar características
            vector<double> features;
            features.insert(features.end(), color_histogram.begin(), color_histogram.end());
            features.insert(features.end(), dominant_colors.begin(), dominant_colors.end());
            features.insert(features.end(), hog_features.begin(), hog_features.end());

            all_features.push_back(features);
        }
    }

    if (all_features.empty()) {
        cerr << "Nenhuma característica extraída. Verifique o diretório e as imagens." << endl;
        return;
    }

    MatrixXd features_matrix(all_features.size(), all_features[0].size());
    for (size_t i = 0; i < all_features.size(); ++i) {
        features_matrix.row(i) = Map<VectorXd>(all_features[i].data(), all_features[i].size());
    }

    Mat features_array = eigen_to_cv(features_matrix);

    auto [labels, cluster_centers] = cluster_images(features_array, 3);

    for (size_t i = 0; i < labels.size(); ++i) {
        cout << "Imagem " << i << " foi atribuída ao cluster " << labels[i] << endl;
    }

    for (size_t i = 0; i < image_filenames.size(); ++i) {
        int cluster_label = labels[i];
        string cluster_dir = output_folder + "/cluster_" + to_string(cluster_label);

        if (!fs::exists(cluster_dir)) {
            fs::create_directories(cluster_dir);
        }

        string src_image_path = image_filenames[i];
        string image_name = fs::path(src_image_path).filename().string();
        string dest_image_path = cluster_dir + "/" + image_name;

        try {
            fs::copy(src_image_path, dest_image_path, fs::copy_options::overwrite_existing);
            cout << "Imagem " << image_name << " foi copiada para " << dest_image_path << endl;
        } catch (const fs::filesystem_error& e) {
            cerr << "Erro ao copiar a imagem " << image_name << ": " << e.what() << endl;
        }
    }
}

int main() {
    string dir_path = "imagens/img210";    // Substitua pelo caminho correto das suas imagens
    vector<string> extensions = { ".jpg", ".jpeg" }; // Extensões das imagens
    string output_folder = "clusterizacao_resultados";  // Substitua pelo caminho de saída

    try {
        load_images_and_extract_features(dir_path, extensions, output_folder);
    } catch (const std::exception& e) {
        cerr << "Erro durante a execução: " << e.what() << endl;
        return 1;
    }

    return 0;
}
