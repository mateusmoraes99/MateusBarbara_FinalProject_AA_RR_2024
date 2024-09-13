import os

# Define o caminho da pasta onde estão as imagens
pasta_imagens = '../imagens/formatura'

# Lista todos os arquivos na pasta
arquivos = os.listdir(pasta_imagens)

# Filtra apenas os arquivos de imagem (ajuste as extensões conforme necessário)
extensoes_imagem = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
imagens = [f for f in arquivos if os.path.splitext(f)[1].lower() in extensoes_imagem]

# Ordena as imagens para garantir que a numeração seja consistente
imagens.sort()

# Renomeia as imagens
for i, imagem in enumerate(imagens):
    # Formata o número com três dígitos
    numero = str(i + 1).zfill(3)
    # Obtém a extensão do arquivo
    _, extensao = os.path.splitext(imagem)
    # Cria o novo nome do arquivo
    novo_nome = f'f_{numero}{extensao}'
    # Cria o caminho completo para o arquivo antigo e o novo arquivo
    caminho_antigo = os.path.join(pasta_imagens, imagem)
    caminho_novo = os.path.join(pasta_imagens, novo_nome)
    # Renomeia o arquivo
    os.rename(caminho_antigo, caminho_novo)

print("Renomeação concluída!")
