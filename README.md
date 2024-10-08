# MateusBarbara_FinalProject_AA_RR_2024

Repositório para o trabalho final da disciplina de Análise de Algoritmos. Trabalho realizado em dupla pelos alunos Bárbara Zamperete e Mateus Moraes.

### Processos e Organização:

- Coleta de imagens de Aniversários, Casamentos e Formaturas (/imagens)
- Agrupamento de variadas quantidades de imagens para realização de testes com diferentes entradas (/img30, /img60, ... /img210)
- Implementação em C++ (main.cpp)
  - Uso do K-means para clusterização
  - Extração de caracteristicas das imagens:
    - Histograma de cores
    - Cor dominante
    - HOG features
  - Separação das imagens em subdiretórios conforme resultado da clusterização
- Codigos auxiliares em Python (/auxiliar)
  - Renomeação e padronização das imagens (renomear_imagens.py)
  - Contagem e análise dos resultados (contagem.py)
- Testes e seus resultados (/testes)
- Criação do relatório (/relatorio e slide/Organização Automatica de Fotografias de Eventos.docx|odt|pdf)
- Criação do slide de apresentação (/relatorio e slide/apresentacao.pptx)
- Taxa de erro solicitada no dia da apresentação 
  - ![Resultado da Clusterização com 210 imagens](contagem_arquivos_clusterizacao_210imgs.png)
  - ![Taxa de erro da Clusterização com 210 imagens](imagens_erradas_por_categoria_210img.png)

### Atividades:

- De 10/09 a 13/09:
  - Coletar imagens para o banco de imagens
  - Estrutura algoritmo k-Means
  - Ler o artigo

- De 14/09 a 18/09:
  - Implementar codigo

## Testes realizados

- 1: Dominant_colors = 3; Histogram bins=16; HOG norm=L2-Hys

- 2: Dominant_colors = 5; Histogram bins=32; HOG norm=L1

- 3: Dominant_colors = 3; Histogram bins=16; HOG norm=L1

- 4: Dominant_colors = 3; Histogram bins=32; HOG norm=L2-Hys

- 5: Dominant_colors = 5; Histogram bins=16; HOG norm=L2-Hys

