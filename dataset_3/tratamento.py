import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# --- CONFIGURAÇÃO INICIAL ---
try:
    stopwords.words('portuguese')
except LookupError:
    print("Baixando a lista de stopwords do NLTK...")
    nltk.download('stopwords')
stop_words_pt = set(stopwords.words('portuguese'))


# --- FUNÇÕES DE LIMPEZA ---

# --- NOVA FUNÇÃO ---
def remover_emojis(texto):
    """Remove emojis do texto."""
    padrao_emoji = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\u2600-\u27BF"          # miscellaneous symbols
        "]+",
        flags=re.UNICODE,
    )
    return padrao_emoji.sub(r'', str(texto))

def remover_urls_e_usernames(texto):
    """Remove padrões de URL e usernames do texto."""
    padrao = re.compile(r'@\w+|https?://\S+|www\.\S+')
    return padrao.sub(r'', str(texto))

def remover_pontuacao(texto):
    """Remove toda a pontuação do texto."""
    return str(texto).translate(str.maketrans('', '', string.punctuation))

def remover_stopwords(texto):
    """Converte para minúsculas e remove as stopwords."""
    texto_lower = str(texto).lower()
    tokens = texto_lower.split()
    texto_sem_stopwords = [palavra for palavra in tokens if palavra not in stop_words_pt]
    return ' '.join(texto_sem_stopwords)


# --- EXECUÇÃO PRINCIPAL ---
arquivo_bruto = r"C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_3\told_apenas_hate_2.0_bruto.csv"
arquivo_tratado_final = "told_apenas_hate_2.0_tratado.csv"

try:
    print(f"Carregando o arquivo '{arquivo_bruto}'...")
    df_bruto = pd.read_csv(arquivo_bruto)
    df_tratado = df_bruto.copy()

    print("\n--- INICIANDO PROCESSO DE LIMPEZA ---")

    # --- NOVO PASSO DE LIMPEZA ---
    print("\n1. Removendo Emojis...")
    df_tratado['text'] = df_tratado['text'].apply(remover_emojis)

    print("2. Removendo URLs e usernames...")
    df_tratado['text'] = df_tratado['text'].apply(remover_urls_e_usernames)

    print("3. Removendo pontuação...")
    df_tratado['text'] = df_tratado['text'].apply(remover_pontuacao)

    print("4. Removendo stopwords e convertendo para minúsculas...")
    df_tratado['text'] = df_tratado['text'].apply(remover_stopwords)

    print("5. Removendo espaços em excesso...")
    df_tratado['text'] = df_tratado['text'].apply(lambda x: " ".join(x.split()))

    df_tratado.to_csv(arquivo_tratado_final, index=False, encoding='utf-8')

    print("\n--- SUCESSO! ---")
    print(f"Arquivo '{arquivo_tratado_final}' foi criado com o texto tratado (sem emojis).")
    print("\nAmostra do arquivo final tratado:")
    print(df_tratado.head().to_string())

except FileNotFoundError:
    print(f"\nERRO: O arquivo '{arquivo_bruto}' não foi encontrado.")
    