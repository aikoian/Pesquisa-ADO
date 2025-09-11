import pandas as pd
import re

# Carrega o dataset 
df = pd.read_csv("dataset_limpo.csv", encoding='utf-8', low_memory=False)

# Selecionar apenas as duas colunas principais (text + hatespech_comb)
df = df[['text', 'hatespeech_comb']].copy()

# Função de limpeza (menções, urls, hashtags, números e caracteres estranhos)

def limpeza_total(texto):
    texto = re.sub(r'@\w+', '', str(texto))               # menções
    texto = re.sub(r'http\S+|www\S+', '', texto)          # URLs
    texto = re.sub(r'#\w+', '', texto)                    # hashtags
    texto = re.sub(r'\d+', '', texto)                     # números
    texto = re.sub(r'[^\w\sáéíóúàâêôãõç]', ' ', texto)    # caracteres não alfanuméricos exceto acentos
    texto = re.sub(r'\s+', ' ', texto)                    # espaços múltiplos
    return texto.strip().lower()

# Aplicando limpeza
df['text'] = df['text'].apply(limpeza_total)

# Removendo textos duplicados
df = df.drop_duplicates(subset='text')
df = df.reset_index(drop=True)

# Salvando o dataset
df.to_csv("dataset_final_enxuto.csv", index=False, encoding='utf-8')

print(df.head(10))
print(df['hatespeech_comb'].value_counts())
