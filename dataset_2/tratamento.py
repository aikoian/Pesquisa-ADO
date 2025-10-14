import nltk
import pandas as pd
from nltk.corpus import stopwords
import re
import string

# Adicionando o download e a criação do set de stopwords aqui no início
try:
    stopwords.words('portuguese')
except LookupError:
    print("Baixando a lista de stopwords do NLTK...")
    nltk.download('stopwords')
stop_words_pt = set(stopwords.words('portuguese'))

def linha():
    print("-="*110)

# --- Funções de Limpeza (mantidas exatamente como as suas) ---

#funcao que remove urls e usernames
def remover_urls_e_usernames(texto):
    """
    Remove padrões de URL (http, https, www) e usernames (@usuario) do texto.
    Esta função deve ser usada ANTES de remover a pontuação.
    """
    # Padrão para encontrar URLs completas ou usernames
    padrao = re.compile(r'@\w+|https?://\S+|www\.\S+')
    return padrao.sub(r'', str(texto))

# funcao que remove stopwords
def remover_stopwords(texto):
    """Converte para minúsculas e remove as stopwords."""
    texto = str(texto).lower()
    tokens = texto.split()
    # Utilizando o set de stopwords criado no início do script
    texto_sem_stopwords = [palavra for palavra in tokens if palavra not in stop_words_pt]
    return ' '.join(texto_sem_stopwords)

# função para retirar pontuação.
def remover_pontuacao(texto):
    """Remove toda a pontuação do texto."""
    return str(texto).translate(str.maketrans('', '', string.punctuation))

# inicio do script principal :

try:
    #? PASSO NOVO: Carregar os dois arquivos necessários.
    df_train = pd.read_csv(r"D:\Pesquisa-ADO\Novo dataset\train.csv")
    df_metadata = pd.read_csv(r"D:\Pesquisa-ADO\Novo dataset\train_metadata.csv")

    #? PASSO NOVO: Unir os dois dataframes pela coluna 'id'
    df = pd.merge(df_train, df_metadata, on="id")

    #? ******** CORREÇÃO PRINCIPAL: REMOVER LINHAS DUPLICADAS ********
    #? Isso garante que cada tweet original apareça apenas uma vez.
    df.drop_duplicates(subset=['id'], inplace=True)


except FileNotFoundError:
    print("O arquivo não pode ser carregado, verifique o caminho para 'train.csv' e 'train_metadata.csv'.")
    exit()
print()
print("-=-=-=-=- Amostra do dataset Original Mesclado -=-=-=-=-")
print(df[['text', 'targeted_type', 'racism', 'sexism']].head())
linha()

# TODO criando uma coluna 'is_hate_speech'
#? A condição agora é muito mais precisa, como você sugeriu.
#? is_hate_speech será 1 apenas se targeted_type for 'GRP' E se for classificado como um tipo de ódio específico.

colunas_de_odio = [
    'racism', 'sexism', 'xenophobia',
    'lgbtqphobia', 'religious_intolerance'
]

condicao_precisa = (df['targeted_type'] == 'GRP') & (df[colunas_de_odio].any(axis=1))
df['is_hate_speech'] = condicao_precisa.astype(int)


print("=-=-= DATASET APÓS CRIAR A COLUNA 'is_hate_speech' (COM LÓGICA PRECISA) =-=-=")
print()
print(f"{df['is_hate_speech'].value_counts()}\nClaramente existe um desbalanço de casos, mas agora com alta precisão.")
print("Exemplos:")
print(df[['text','targeted_type', 'racism', 'sexism','is_hate_speech']].head(10))
linha()

# separando oque vamos usar (sem alteração aqui)
df_final = df[['text','is_hate_speech']].copy()
df_final.dropna(subset=['text'], inplace=True)

#! APLICANDO O PRÉ-PROCESSAMENTO... (sem alteração na sua lógica)
linha()
print("1 - Removendo URLs e usernames...")
print()
df_final['text'] = df_final['text'].apply(remover_urls_e_usernames)
print(df_final.head(3))
linha()

print("2 - Removendo pontuação...")
print()
df_final['text'] = df_final['text'].apply(remover_pontuacao)
print(df_final.head(3))
linha()
print()
print("3 - Removendo stopwords e convertendo para minúsculas...")
print()
df_final['text'] = df_final['text'].apply(remover_stopwords)
print(df_final.head(3)) # Adicionado para visualização
linha()
print()
print("4 - Removendo espaços em excesso...")
df_final['text'] = df_final['text'].apply(lambda x: " ".join(x.split()))
print()
linha()
print()

# FINALIZAÇÃO
linha()
print("--- Amostra do data set final ---")
print()
print(df_final.head(10))

#? Salvar o resultado final em um novo arquivo.
output_filename = "dataset_final_precisao.csv"
df_final.to_csv(output_filename, index=False, encoding='utf-8')
print(f"\nArquivo final com filtragem precisa e sem duplicatas salvo como: '{output_filename}'")