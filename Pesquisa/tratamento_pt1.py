import nltk
import pandas as pd


nltk.download('punkt')     # -> tokenização
nltk.download('stopwords') # -> stopwords
nltk.download('rslp')      # -> stemmer para português

arquivo = 'Dataset.csv'
df = pd.read_csv(arquivo)

print("Colunas do dataset:")

print(df.columns)

print("\n Primeiras linhas: ")

print(df.head())

print("\n Informações do dataset:")
print(df.info())

df_final = df[['text', 'hatespeech_comb']]

print(df_final.columns)
print(df_final.head)
