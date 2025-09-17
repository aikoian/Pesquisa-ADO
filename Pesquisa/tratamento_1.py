import nltk
from nltk.corpus import stopwords
import re 
import pandas as pd
import string

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Workspace\estudos\Pesquisa-ADO\Dataset.csv", encoding='utf-8', low_memory=False)

# Selecionar apenas as duas colunas principais (text + hatespech_comb)
df = df[['text', 'hatespeech_comb']].copy()

stop_words = set(stopwords.words('portuguese'))

def remover_stopwords(texto):
    tokens = texto.split()
    texto_sem_stopwords = [palavra for palavra in tokens if palavra not in stop_words]
    return ' '.join(texto_sem_stopwords)

df['text'] = df['text'].apply(remover_stopwords)
print(df.head(10))



# Esse tratamento está focando em retirar espaços em excesso e pontuação.

# print("Informações do dataset: ")
# print(df.info)

# print("\nPrimeiros 5 registros")
# print(df.head)

# Remover a pontuação

def remover_pontuacao(texto):
    return texto.translate(str.maketrans('','',string.punctuation))

df['text'] = df['text'].astype(str).apply(remover_pontuacao)

# Remover espaços em excesso

df['text'] = df['text'].apply(lambda x: " ".join(x.split()))

# print("\nPrimeiros 5 registros após a remoção de pontuação: ")
# print(df.head())

df.to_csv("dataset_tratado_final.csv", index=False, encoding='utf-8')