import os
import matplotlib.pyplot as plt

# Caminho do diretório principal
diretorio_principal = 'imagens/clusterizacao_resultados'

# Extensões comuns de imagens
extensoes_imagens = ('.jpg', '.jpeg', '.png', '.webp')

# Listas para armazenar os dados para o gráfico
subdiretorios = []
contagens_a = []
contagens_c = []
contagens_f = []

# Percorrer os subdiretórios dentro de 'clusterizacao_resultado'
for subdiretorio in os.listdir(diretorio_principal):
    caminho_subdiretorio = os.path.join(diretorio_principal, subdiretorio)
    
    # Verificar se é um diretório
    if os.path.isdir(caminho_subdiretorio):
        # Contadores para os arquivos que começam com 'a', 'c' e 'f'
        contagem_a = 0
        contagem_c = 0
        contagem_f = 0

        # Percorrer todos os arquivos no subdiretório
        for arquivo in os.listdir(caminho_subdiretorio):
            # Verificar se o arquivo é uma imagem
            if arquivo.lower().endswith(extensoes_imagens):
                # Contar quantos começam com 'a', 'c' ou 'f'
                if arquivo.lower().startswith('a'):
                    contagem_a += 1
                elif arquivo.lower().startswith('c'):
                    contagem_c += 1
                elif arquivo.lower().startswith('f'):
                    contagem_f += 1

        # Adicionar as contagens e o nome do subdiretório às listas
        subdiretorios.append(subdiretorio)
        contagens_a.append(contagem_a)
        contagens_c.append(contagem_c)
        contagens_f.append(contagem_f)

# Gerar o gráfico de barras
fig, ax = plt.subplots(figsize=(10, 6))
largura_barra = 0.2
indice_barras = range(len(subdiretorios))

# Barras para cada contagem
ax.bar([i - largura_barra for i in indice_barras], contagens_a, width=largura_barra, label='Aniversário (a)', color='blue')
ax.bar(indice_barras, contagens_c, width=largura_barra, label='Casamento (c)', color='green')
ax.bar([i + largura_barra for i in indice_barras], contagens_f, width=largura_barra, label='Formatura (f)', color='red')

# Adicionar rótulos e título
ax.set_xlabel('Subdiretórios')
ax.set_ylabel('Quantidade de Arquivos')
ax.set_title('Resultado da Clusterização')
ax.set_xticks(indice_barras)
ax.set_xticklabels(subdiretorios, rotation=45, ha='right')

# Legenda
ax.legend()

# Ajustar layout para evitar sobreposição
plt.tight_layout()

# Salvar o gráfico como imagem
plt.savefig('contagem_arquivos_clusterizacao.png')

