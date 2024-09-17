#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <Eigen/Dense>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;
using namespace Eigen;


Mat resize_image(const Mat& image, Size size = Size(128, 128)) {
    Mat resized_image;
    resize(image, resized_image, size);
    return resized_image;
}

vector<double> extract_dominant_colors(const Mat& image, int k = 5) {
    Mat rgb_image;
    cvtColor(image, rgb_image, COLOR_BGR2RGB);
    
    Mat samples(rgb_image.rows * rgb_image.cols, 3, CV_32F);
    for (int y = 0; y < rgb_image.rows; ++y) {
        for (int x = 0; x < rgb_image.cols; ++x) {
            Vec3b color = rgb_image.at<Vec3b>(y, x);
            samples.at<float>(y * rgb_image.cols + x, 0) = color[0];
            samples.at<float>(y * rgb_image.cols + x, 1) = color[1];
            samples.at<float>(y * rgb_image.cols + x, 2) = color[2];
        }
    }

    Mat labels;
    Mat centers;
    kmeans(samples, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2), 3, KMEANS_RANDOM_CENTERS, centers);

    vector<double> colors;
    for (int i = 0; i < k; ++i) {
        colors.push_back(centers.at<float>(i, 0));
        colors.push_back(centers.at<float>(i, 1));
        colors.push_back(centers.at<float>(i, 2));
    }
    return colors;
}

vector<double> extract_color_histogram(const Mat& image, const vector<int>& bins = {16, 16, 16}) {
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_BGR2HSV);

    vector<Mat> hsv_planes;
    split(hsv_image, hsv_planes);

    Mat hist;
    int histSize[] = { bins[0], bins[1], bins[2] };
    float h_ranges[] = { 0, 256 };
    float s_ranges[] = { 0, 256 };
    float v_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges, v_ranges };
    int channels[] = { 0, 1, 2 };
    calcHist(hsv_planes.data(), 3, channels, Mat(), hist, 3, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX);

    vector<double> hist_vector(hist.begin<float>(), hist.end<float>());
    return hist_vector;
}

vector<double> extract_hog_features(const Mat& image) {
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    vector<double> features;
    HOGDescriptor hog;
    hog.blockSize = Size(16, 16);
    hog.cellSize = Size(8, 8);
    hog.blockStride = Size(8, 8);
    hog.nbins = 9;

    vector<float> descriptors;
    hog.compute(gray_image, descriptors);

    features.assign(descriptors.begin(), descriptors.end());
    return features;
}

pair<vector<int>, Mat> cluster_images(const Mat& features, int num_clusters = 3) {
    Mat labels;
    Mat centers;
    
    kmeans(features, num_clusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2), 3, KMEANS_RANDOM_CENTERS, centers);

    vector<int> labels_vector(labels.begin<int>(), labels.end<int>());
    return { labels_vector, centers };
}

void create_cluster_folders(const string& output_folder, int num_clusters) {
    for (int i = 0; i < num_clusters; ++i) {
        fs::path cluster_folder = fs::path(output_folder) / ("cluster_" + to_string(i));
        if (!fs::exists(cluster_folder)) {
            fs::create_directory(cluster_folder);
        }
    }
}

void copy_images_to_clusters(const vector<string>& image_paths, const vector<int>& labels, const string& output_folder) {
    int num_clusters = *max_element(labels.begin(), labels.end()) + 1;
    create_cluster_folders(output_folder, num_clusters);

    for (size_t i = 0; i < image_paths.size(); ++i) {
        fs::path src = image_paths[i];
        fs::path dst = fs::path(output_folder) / ("cluster_" + to_string(labels[i])) / src.filename();
        fs::copy(src, dst);
    }
}

void load_images_and_extract_features(const string& image_folder, const string& output_folder) {
    vector<string> image_paths;
    for (const auto& entry : fs::directory_iterator(image_folder)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
            image_paths.push_back(entry.path().string());
        }
    }
    
    if (image_paths.empty()) {
        throw runtime_error("Nenhuma imagem foi encontrada na pasta especificada.");
    }
    
    vector<vector<double>> features_list;
    for (const auto& image_path : image_paths) {
        cout << "Carregando imagem: " << image_path << endl;
        Mat image = imread(image_path);
        if (!image.empty()) {
            Mat resized_image = resize_image(image);
            vector<double> dominant_colors = extract_dominant_colors(resized_image);
            vector<double> hist = extract_color_histogram(resized_image);
            vector<double> hog_features = extract_hog_features(resized_image);

            vector<double> combined_features;
            combined_features.insert(combined_features.end(), dominant_colors.begin(), dominant_colors.end());
            combined_features.insert(combined_features.end(), hist.begin(), hist.end());
            combined_features.insert(combined_features.end(), hog_features.begin(), hog_features.end());
            features_list.push_back(combined_features);
        } else {
            cout << "Erro ao carregar imagem: " << image_path << endl;
        }
    }

    MatrixXd features_array(features_list.size(), features_list[0].size());
    for (size_t i = 0; i < features_list.size(); ++i) {
        for (size_t j = 0; j < features_list[i].size(); ++j) {
            features_array(i, j) = features_list[i][j];
        }
    }

    auto [labels, cluster_centers] = cluster_images(features_array, 3);

    // Função de plotagem omitida por simplicidade
    // plot_cluster_distribution(features_array, labels, "cluster_distribution.png");

    copy_images_to_clusters(image_paths, labels, output_folder);

    cout << "Clusterização concluída, imagens copiadas para pastas dos clusters." << endl;
}

int main() {
    string image_folder = "imagens/todas_imagens";
    string output_folder = "clusterizacao_resultados";
    if (!fs::exists(output_folder)) {
        fs::create_directory(output_folder);
    }

    try {
        load_images_and_extract_features(image_folder, output_folder);
    } catch (const exception& e) {
        cerr << "Erro: " << e.what() << endl;
        return 1;
    }

    return 0;
}

