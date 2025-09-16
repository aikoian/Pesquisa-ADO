import nltk
from nltk.corpus import stopwords
import re 
import pandas as pd
nltk.download('stopwords')
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

df.to_csv("dataset_tratado_1",index = False, encoding='utf-8')
