## Análise da Complexidade do Codigo


##### Função main()

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

Todas as operações tem complexidade O(1) com excessão da chamada da função `load_images_and_extract_features`.


##### Função load_images_and_extract_features()

Primeira parte: O(1)

    void load_images_and_extract_features(const string& dir_path, const vector<string>& extensions, const string& output_folder) {
        vector<Mat> images;
        vector<vector<double>> all_features;
        vector<string> image_filenames;

        Size fixed_size(128, 128); // Tamanho fixo para redimensionamento

Segunda parte: 
- Iteração sobre cada imagem O(n)


        for (const auto& entry : fs::directory_iterator(dir_path)) {
            string ext = entry.path().extension().string();

- Para cada imagem é feito:
    - Leitura da imagem: O(WxH), onde W e H são a altura e largura da imagem

            if (find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                cout << "Carregando imagem: " << entry.path().string() << endl;
                Mat image = imread(entry.path().string());

                if (image.empty()) {
                    cerr << "Falha ao carregar imagem: " << entry.path().string() << endl;
                    continue;
                }

    - Redimensionamento da imagem: O(WxH)

                Mat resized_image = resize_image(image, fixed_size);

                images.push_back(resized_image);
                image_filenames.push_back(entry.path().string());

    - Extrair características

        - Histograma de Cores (A):

                vector<double> color_histogram = extract_color_histogram(resized_image);

        - Extração da Cor Dominante (B):

                vector<double> dominant_colors = extract_dominant_colors(resized_image);
    
        - Extração de HOG Features (C):

                vector<double> hog_features = extract_hog_features(resized_image);

    - Concatenasr as features extraidas nas funções anteriores: O(F) onde F é a quantidade de features.

                vector<double> features;
                features.insert(features.end(), color_histogram.begin(), color_histogram.end());
                features.insert(features.end(), dominant_colors.begin(), dominant_colors.end());
                features.insert(features.end(), hog_features.begin(), hog_features.end());

                all_features.push_back(features);
            }
        }

- Terceira parte : Ao fim do loop para extrair redimensionar e extrair caracteristicas de todas as imagens, é formado a Matriz de Features. O(N x F).

        MatrixXd features_matrix(all_features.size(), all_features[0].size());
        for (size_t i = 0; i < all_features.size(); ++i) {
            features_matrix.row(i) = Map<VectorXd>(all_features[i].data(), all_features[i].size());
        }

        Mat features_array = eigen_to_cv(features_matrix);

- Quarta parte: Clusterização (D)

        auto [labels, cluster_centers] = cluster_images(features_array, 3);

- Quinta parte: Copiar e separar as imagens em diretorios conforme o resultado da clusterização.

    - Isso é feito para cada imagem N. A função de `copy()` tem custo O(WxH). Totalizando O(N x W x H);

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
            }}}
            