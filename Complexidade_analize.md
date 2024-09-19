## Análise da Complexidade do Codigo


##### Função main()

```
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
```

Todas as operações tem complexidade O(1) com excessão da chamada da função `load_images_and_extract_features`.


##### Função load_images_and_extract_features()

Primeira parte: O(1)

```
    void load_images_and_extract_features(const string& dir_path, const vector<string>& extensions, const string& output_folder) {
        vector<Mat> images;
        vector<vector<double>> all_features;
        vector<string> image_filenames;

        Size fixed_size(128, 128); // Tamanho fixo para redimensionamento
```

Segunda parte: 
- Iteração sobre cada imagem O(n)


        for (const auto& entry : fs::directory_iterator(dir_path)) {
            string ext = entry.path().extension().string();

- Para cada imagem é feito:
    - Leitura da imagem: O(WxH), onde W e H são a altura e largura da imagem
    ```

        if (find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
            cout << "Carregando imagem: " << entry.path().string() << endl;
            Mat image = imread(entry.path().string());

            if (image.empty()) {
                cerr << "Falha ao carregar imagem: " << entry.path().string() << endl;
                continue;
            }
    ```

    - Redimensionamento da imagem: O(WxH)

    ```
        Mat resized_image = resize_image(image, fixed_size);

        images.push_back(resized_image);
        image_filenames.push_back(entry.path().string());
    ```

    - Extrair características

        - **Histograma de Cores (A):**
        Essa função tem a complexidade constante, já que as imagens passadas estão redimensionadas para um tamanho fixo. **O(128x128)**.

                vector<double> color_histogram = extract_color_histogram(resized_image);

        - **Extração da Cor Dominante (B):**
        Essa função tem a complexidade constante, já que as imagens passadas estão redimensionadas para um tamanho fixo e a quantidade de cores extraida também é fixa. **O(128x128x5)**.

                vector<double> dominant_colors = extract_dominant_colors(resized_image);
    
        - **Extração de HOG Features (C):**
        Essa função também tem coplexidade constante, pois depende do tamanho das imagens que é um valor fixo. **O(128x128)**.

                vector<double> hog_features = extract_hog_features(resized_image);


    - Concatenar as features extraidas nas funções anteriores: O(F) onde F é a quantidade de features.

```
            vector<double> features;
            features.insert(features.end(), color_histogram.begin(), color_histogram.end());
            features.insert(features.end(), dominant_colors.begin(), dominant_colors.end());
            features.insert(features.end(), hog_features.begin(), hog_features.end());

            all_features.push_back(features);
        }
    }
```

> A complexidade total da segunda parte é O(N x H x W + N x F) = O(N x (H x W + F))

- Terceira parte : Ao fim do loop para extrair redimensionar e extrair caracteristicas de todas as imagens, é formado a Matriz de Features. O(N x F).

```
    MatrixXd features_matrix(all_features.size(), all_features[0].size());
    for (size_t i = 0; i < all_features.size(); ++i) {
        features_matrix.row(i) = Map<VectorXd>(all_features[i].data(), all_features[i].size());
    }

    Mat features_array = eigen_to_cv(features_matrix);
```
> A complexidade total da terceira parte é O(N x F)

- Quarta parte: **Clusterização (D)**
A clusterização é calculada em cima do numero de imagens (N), o numero de features (F) e o número de clusters (k). Como k é um valor constante (k=3), a complexidada da clusterização se reduz a **O(NxF).**

```
    auto [labels, cluster_centers] = cluster_images(features_array, 3);
```

> A complexidade total da quarta parte é O(N x F)

- Quinta parte: Copiar e separar as imagens em diretorios conforme o resultado da clusterização.

    - Isso é feito para cada imagem N. A função de `copy()` tem custo O(WxH). Totalizando **O(N x W x H)** onde W e H representam as dimensões originais das imagens.

```
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
```           

> A complexidade total da quinta parte é O(N x H x W)


### Complexidade total:

A complexidade total do algoritmo é dada, portante a partir da soma das complexidades de cada parte.

- Primeira: O(1)
- Segunda: O(N x H x W + N x F)
- Terceira: O(N x F)
- Quarta: O(N x F)
- Quinta: O(N x H x W)

**Complexidade total: O( N ( H x W + F))**
Onde:
- N: quantidade de imagens
- H x W: as dimensões das imagens
- F: a quantidade de features extraida

 Essa expressão reflete que o tempo de execução do algoritmo depende tanto do número de imagens quanto das dimensões das imagens e da quantidade de características extraídas.


### Detalhamento da análise das funções (A), (B), (C) e (D).

- (A) Função extract_color_histogram
- (B) Função extract_dominant_colors
- (C) Função extract_hog_features
- (D) Clusterização


##### (A) Função extract_color_histogram()

Essa função recebe as imagens já redimensionadas para o valor fixo de 128x128.

- A chamada da função `cvtColor` converte a imagem para HSV, isso é feito analisando para pixel da imagem, o que é uma operação constante. O(128x128)

```
    vector<double> extract_color_histogram(const Mat& image, int bins = 32) { 
        Mat hsv_image;
        cvtColor(image, hsv_image, COLOR_BGR2HSV);
```

- As proximas operações são de atribuição, portanto com a complexidade O(1)

```
    int histSize[] = { bins, bins, bins };
        float h_ranges[] = { 0, 180 };  // Hues em OpenCV vão de 0 a 180
        float s_ranges[] = { 0, 256 };
        float v_ranges[] = { 0, 256 };
        const float* ranges[] = { h_ranges, s_ranges, v_ranges };
        int channels[] = { 0, 1, 2 };

        Mat hist;
```

- Em seguida é utilizada a função `calcHist` para o calculo do histograma, que é calculado em cima de 3 canais de `bins`. Os bins foram definidos com o valor 32. O(32³)

        calcHist(&hsv_image, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);

- Por fim ocorre a função de normalização, que também é constante e é calculado em cima da quantidade de bins. O(32³)

        normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

Como o termo dominante é O(H x W) = O(128 x 128), logo
> **A complexidade total da função (A) é constante: O(128 x 128)**


##### (B) Função extract_dominant_colors()

Essa função recebe as imagens já redimensionadas para o valor fixo de 128x128.

- A chamada da função `cvtColor` converte a imagem para HSV, isso é feito analisando para pixel da imagem, o que é uma operação constante. O(128x128)

```
    vector<double> extract_dominant_colors(const Mat& image, int k = 5) {  
        Mat image_rgb;
        cvtColor(image, image_rgb, COLOR_BGR2RGB);
```

- Em seguida a imagem é redimensionada para uma matriz de pixels e os dados são convertidos para `CV_32F`. O(128x128)

```
    Mat pixels = image_rgb.reshape(1, image_rgb.total());
    pixels.convertTo(pixels, CV_32F);
```

- Aplicação do K-means para identificação das cores dominantes. O tempo de execução do K-means depende do tamanho da imagem(128x128), a quantidade de clusters (quantidade de cores dominantes que são extraidas k=5) e o número máximo de iterações (i) que no pior caso é constante para toda as imagens. O(128x128x5xi) é constante.

```
    Mat labels, centers;
    kmeans(pixels, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2), 3, KMEANS_PP_CENTERS, centers);
```

- Em seguida o loop a baixo é calculado para cada cor k, sendo k=5 e as operações dentro do loop constantes, a complexidade desse trecho é O(k), como k é fixo e pequeno a complexidade é constante.

```
    vector<double> dominant_colors;
        for (int i = 0; i < k; ++i) {
            Vec3f color = centers.at<Vec3f>(i, 0);
            dominant_colors.push_back(color[0]); // R
            dominant_colors.push_back(color[1]); // G
            dominant_colors.push_back(color[2]); // B
        }

        return dominant_colors;
```

> **Portanto a complexidade total da função (B) é constante: O(128x128x5)**

##### (C) Função extract_hog_features()

- Essa função é calculada com base nas dimensões da imagens passada como entrada, como esse valor é fixo (128x128) o valor da complexidade dessa função também será constante.

```
vector<double> extract_hog_features(const Mat& image) {
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    Size winSize(64, 128);  
    Size blockSize(16, 16); 
    Size blockStride(8, 8); 
    Size cellSize(8, 8);    

    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, 9, 1, -1, HOGDescriptor::L2Hys, 0.2);

    vector<float> descriptors;
    hog.compute(gray_image, descriptors, Size(8, 8), Size(0, 0));

    return vector<double>(descriptors.begin(), descriptors.end());
}
```
> **Portanto a complexidade total da função (C) é constante O(128x128)**

##### (D) Clusterizacao

Essa função recebe as imagens já redimensionadas para o valor fixo de 128x128.

- A chamada da função `cvtColor` converte a imagem para HSV, isso é feito analisando para pixel da imagem, o que é uma operação constante. O(128x128)

```
pair<vector<int>, Mat> cluster_images(const Mat& features, int num_clusters = 3) {
    Mat features_32f;
    features.convertTo(features_32f, CV_32F);
```

- Ocorre então a normalização das caracteristicas, que é feita iterando sobre os elementos da matriz. Como a matriz tem as dimensões associadas a quantidade de imagens e de features, O(N x F)

```
Mat normalized_features = normalize_features(features_32f);
```

- É aplicado o K-means para clusterizar as imagens com base em suas features. A complexidade do k-means depende do número de imagens (N), quantidade de caracteristicas por imagens (F), número de cluster (k=3) e o número de iterações (T) que é limitado a T<=100.
- Portanto a complexidade do k-means é O(NxF), já que k é constante e T é limitado a 100.

```
    Mat labels, centers;
    kmeans(normalized_features, num_clusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2), 3, KMEANS_PP_CENTERS, centers);
```

- Por fim a matriz de labels é convertida para um vetor, que varia em relação a quantidade de imagens N. Portanto O(N).

```
    vector<int> labels_vec;
    labels_vec.assign((int*)labels.datastart, (int*)labels.dataend);

    return { labels_vec, centers };
```

> **Portanto a complexidade total da função (D) dada por O(NxF)**
