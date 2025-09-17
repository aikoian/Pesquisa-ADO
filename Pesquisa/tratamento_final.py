import nltk
from nltk.corpus import stopwords
import re 
import pandas as pd
import string

# funcao que remove stopwords
def remover_stopwords(texto):
    tokens = texto.split()
    texto_sem_stopwords = [palavra for palavra in tokens if palavra not in stop_words]
    return ' '.join(texto_sem_stopwords)


# a partir daqui, esse tratamento está focando em retirar espaços em excesso e pontuação.
def remover_pontuacao(texto):
    return texto.translate(str.maketrans('','',string.punctuation))

#aqui se lê o arquivo do dataset original
df = pd.read_csv(r"C:\Users\guilb_3cws35i\OneDrive\Documentos\GitHub\Pesquisa-ADO\Dataset.csv", encoding='utf-8', low_memory=False)

# Selecionar apenas as duas colunas principais (text + hatespech_comb)
df = df[['text', 'hatespeech_comb']].copy()

# essa linha pega todas as stopword portuguesas ("o", "a", "do"...) e transforma elas num set (tipo de variavel mais rapido de fazer buscas)
stop_words = set(stopwords.words('portuguese'))

df['text'] = df['text'].apply(remover_stopwords)

df['text'] = df['text'].astype(str).apply(remover_pontuacao)

# Remover espaços em excesso
df['text'] = df['text'].apply(lambda x: " ".join(x.split()))

print(df.head(10))

# df.to_csv é o comando do Pandas para salvar um DataFrame em um arquivo CSV.
df.to_csv("dataset_tratado_final.csv", index=False, encoding='utf-8')