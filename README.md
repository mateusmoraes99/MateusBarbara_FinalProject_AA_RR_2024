# MateusBarbara_FinalProject_AA_RR_2024


[DESCRIÇÃO - 12] Clusterização
• Faça um análise e descrição do seguinte artigo: T. N. Pappas and N. S. Jayant, "An adaptive
clustering algorithm for image segmentation," International Conference on Acoustics,
Speech, and Signal Processing,, Glasgow, UK, 1989, pp. 1667-1670 vol.3, doi:
10.1109/ICASSP.1989.266767.
• Implemente uma solução para Organização de Fotografias de Eventos
Contexto: Você possui uma grande coleção de fotos tiradas em diferentes eventos ao longo de um
ano. Quer organizar essas fotos em categorias, como "Aniversário", "Férias", "Casamento", etc.,
usando clustering para agrupar imagens similares sem a necessidade de treinar um modelo de
aprendizado de máquina.
Objetivo: Agrupar fotos em diferentes categorias com base em características visuais semelhantes,
usando um algoritmo de clustering, como K-Means ou DBSCAN, para facilitar a organização e a
busca das imagens.
Passos para Implementação:
Preparação dos Dados:
• Coleta de Imagens: Reúna todas as fotos em uma pasta específica.
• Pré-processamento: Reduza o tamanho das imagens, se necessário, para otimizar o
processamento.
Extração de Recursos:
• Utilize bibliotecas como OpenCV ou PIL para carregar as imagens e converter para
um formato que pode ser utilizado pelo algoritmo de clustering.
• Extraia características relevantes das imagens, como histogramas de cores,
descritores de bordas, ou outros vetores de características. Pode-se usar métodos
como o Histograma de Cores ou o Histogram of Oriented Gradients (HOG) para isso.
Aplicação do Algoritmo de Clustering:
• Escolha do Algoritmo: Utilize o K-Means ou DBSCAN do scikit-learn. O K-
Means pode ser uma escolha simples para começar, onde você define o número de
clusters com base em uma estimativa inicial.
Universidade Federal de Roraima
Departamento de Ciência da Computação
Análise de Algoritmos – DCC606
Análise e Organização:
• Classificação: Use os rótulos gerados pelo clustering para organizar as imagens em
pastas ou categorias. As imagens que compartilham o mesmo rótulo serão agrupadas
na mesma categoria.
Verificação e Ajustes:
• Revise os agrupamentos e ajuste o número de clusters ou parâmetros do algoritmo,
se necessário, para melhorar a precisão dos agrupamentos.
