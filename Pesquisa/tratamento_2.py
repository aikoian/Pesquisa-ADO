# Esse tratamento está focando em retirar espaços em excesso e pontuação.

import pandas as pd
import re
from nltk.corpus import stopwords
import string

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Workspace\estudos\Pesquisa-ADO\dataset_tratado_1.csv")

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

df.to_csv("dataset_tratado_2.csv", index=False, encoding='utf-8')