import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import string

nltk.download('stopwords')

# --- Funções de Limpeza ---

#funcao que remove urls e usernames
def remover_urls_e_usernames(texto):
    """
    Remove padrões de URL (http, https, www) e usernames (@usuario) do texto.
    Esta função deve ser usada ANTES de remover a pontuação.
    """
    # Padrão para encontrar URLs completas ou usernames
    padrao = re.compile(r'@\w+|https?://\S+|www\.\S+')
    return padrao.sub(r'', texto)

# funcao que remove stopwords
def remover_stopwords(texto):
    """Converte para minúsculas e remove as stopwords."""
    texto = texto.lower()
    tokens = texto.split()
    texto_sem_stopwords = [palavra for palavra in tokens if palavra not in stop_words]
    return ' '.join(texto_sem_stopwords)

# função para retirar pontuação.
def remover_pontuacao(texto):
    """Remove toda a pontuação do texto."""
    return texto.translate(str.maketrans('', '', string.punctuation))

# --- Início do Script Principal ---

# Lê o arquivo do dataset original (texto bruto)
df = pd.read_csv(r"C:\Users\guilb_3cws35i\OneDrive\Documentos\GitHub\Pesquisa-ADO\dataset_original.csv", encoding='utf-8', low_memory=False)

# Selecionar apenas as duas colunas principais (text + hatespech_comb)
df = df[['text', 'hatespeech_comb']].copy()

# essa linha pega todas as stopword portuguesas ("o", "a", "do"...) e transforma elas num set (tipo de variavel mais rapido de fazer buscas)
stop_words = set(stopwords.words('portuguese'))


# --- FLUXO DE TRATAMENTO ---

# PASSO 1: Remove URLs e usernames do texto original (ainda com pontuação)
df['text'] = df['text'].astype(str).apply(remover_urls_e_usernames)

# PASSO 2: Agora sim, remove o resto da pontuação
df['text'] = df['text'].apply(remover_pontuacao)

# PASSO 3: Remove stopwords
df['text'] = df['text'].apply(remover_stopwords)

# PASSO 4: Remove espaços em excesso
print("Passo 4: Limpando espaços extras...")
df['text'] = df['text'].apply(lambda x: " ".join(x.split()))


# Testa no terminal as 10 primeiras linhas do dataset tratado
print(df.head(10))

# Salva o DataFrame no arquivo final
df.to_csv("dataset_final.csv", index=False, encoding='utf-8')