# Importação de bibliotecas

import nltk
import pandas as pd
import re
#----------------------------

# Função para remoção do Nome dos Usuários

def remove_mencoes(texto):
    return re.sub(r'@\w+','', texto)
#----------------------------

# Download dos recursos]

# nltk.download('punkt')     # -> tokenização
# nltk.download('stopwords') # -> stopwords
# nltk.download('rslp')      # -> stemmer para português
#----------------------------

# Carregando o dataset original

arquivo = 'Dataset.csv'
df = pd.read_csv(arquivo)
#----------------------------

# Filtro de quais colunas vão ser utilizadas -> o comentário + rótulo geral

df_final = df[['text', 'hatespeech_comb']]
#----------------------------

# Cria uma coluna com o texto limpo (Sem nome dos usuários)

df_final['text_limpo'] = df_final['text'].apply(remove_mencoes)
#----------------------------

# Visualização das primeiras 10 linhas para comparação

print(df_final[['text','text_limpo']].head(10))
#----------------------------




