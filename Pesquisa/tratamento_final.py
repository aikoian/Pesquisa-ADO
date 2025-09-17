import nltk
from nltk.corpus import stopwords
import re 
import pandas as pd
import string

# nltk.download('stopwords')

# funcao que remove stopwords
def remover_stopwords(texto):
    tokens = texto.split()
    texto_sem_stopwords = [palavra for palavra in tokens if palavra not in stop_words]
    return ' '.join(texto_sem_stopwords)

# função para retirar pontuação.
def remover_pontuacao(texto):
    return texto.translate(str.maketrans('','',string.punctuation))

# aqui se lê o arquivo do dataset original
df = pd.read_csv(r"C:\Users\guilb_3cws35i\OneDrive\Documentos\GitHub\Pesquisa-ADO\dataset_original.csv", encoding='utf-8', low_memory=False)

# Selecionar apenas as duas colunas principais (text + hatespech_comb)
df = df[['text', 'hatespeech_comb']].copy()

# essa linha pega todas as stopword portuguesas ("o", "a", "do"...) e transforma elas num set (tipo de variavel mais rapido de fazer buscas)
stop_words = set(stopwords.words('portuguese'))

# usa a def remover_stopwords pra tirar as stopwords
df['text'] = df['text'].apply(remover_stopwords)

# usa a def remover_pontuacao pra tirar a pontuação
df['text'] = df['text'].astype(str).apply(remover_pontuacao)

# Remover espaços em excesso
df['text'] = df['text'].apply(lambda x: " ".join(x.split()))

# testa no terminal as 10 primeiras linhas do dataset tratado
print(df.head(10))

# df.to_csv é o comando do Pandas para salvar um DataFrame em um arquivo CSV.
df.to_csv("dataset_final.csv", index=False, encoding='utf-8')