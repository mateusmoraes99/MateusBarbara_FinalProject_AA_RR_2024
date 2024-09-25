import os
import matplotlib.pyplot as plt

# Caminho do diretório principal
diretorio_principal = 'clusterizacao_resultados'

# Extensões comuns de imagens
extensoes_imagens = ('.jpg', '.jpeg', '.png', '.webp')

# Listas para armazenar os dados para o gráfico
subdiretorios = []
contagens_a = []
contagens_c = []
contagens_f = []
contagens_erradas_a = []  # Para contar imagens erradas de aniversário
contagens_erradas_c = []  # Para contar imagens erradas de casamento
contagens_erradas_f = []  # Para contar imagens erradas de formatura

# Conjunto para rastrear categorias predominantes
categorias_predominantes = set()

# Percorrer os subdiretórios dentro de 'clusterizacao_resultados'
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

        # Exibir a quantidade de imagens de cada categoria no terminal
        print(f"Cluster: {subdiretorio} - Aniversário: {contagem_a}, Casamento: {contagem_c}, Formatura: {contagem_f}")

        # Determinar a categoria predominante
        total = contagem_a + contagem_c + contagem_f
        if total > 0:
            # Encontrar as categorias com a maior contagem
            max_count = max(contagem_a, contagem_c, contagem_f)
            candidatas = []
            if contagem_a == max_count:
                candidatas.append('a')
            if contagem_c == max_count:
                candidatas.append('c')
            if contagem_f == max_count:
                candidatas.append('f')

            # Resolver empates considerando categorias que não foram predominantes
            predominante = None
            for candidata in candidatas:
                if candidata not in categorias_predominantes:
                    predominante = candidata
                    break
            
            # Se não houver categorias não utilizadas, usar a primeira do empate
            if predominante is None and candidatas:
                predominante = candidatas[0]

            # Adicionar a categoria predominante ao conjunto
            categorias_predominantes.add(predominante)

            # Calcular quantas imagens estão no cluster errado
            erradas_a = contagem_a if predominante != 'a' else 0
            erradas_c = contagem_c if predominante != 'c' else 0
            erradas_f = contagem_f if predominante != 'f' else 0

            contagens_erradas_a.append(erradas_a)
            contagens_erradas_c.append(erradas_c)
            contagens_erradas_f.append(erradas_f)
        else:
            contagens_erradas_a.append(0)
            contagens_erradas_c.append(0)
            contagens_erradas_f.append(0)

# Gerar o gráfico de barras para a contagem de imagens
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

# Gerar gráfico de pizza para mostrar erros em porcentagem
fig2, axs = plt.subplots(1, len(subdiretorios), figsize=(15, 5), sharey=True)

for i, subdiretorio in enumerate(subdiretorios):
    erros = [contagens_erradas_a[i], contagens_erradas_c[i], contagens_erradas_f[i]]
    labels = ['Aniversário Erradas', 'Casamento Erradas', 'Formatura Erradas']
    
    # Apenas plotar se houver erros
    if sum(erros) > 0:
        axs[i].pie(erros, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
        axs[i].axis('equal')  # Igualar o aspecto do gráfico para um círculo
        axs[i].set_title(f'Erros em {subdiretorio}')
    else:
        axs[i].pie([1], labels=['Sem Erros'], colors=['lightgray'])  # Gráfico de pizza vazio
        axs[i].set_title(f'Erros em {subdiretorio}')
        
# Ajustar layout
plt.tight_layout()

# Salvar todos os gráficos de pizza em uma única imagem
plt.savefig('imagens_erradas_por_categoria.png')
plt.show()  # Mostrar todos os gráficos
