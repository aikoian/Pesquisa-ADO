import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# --- CONFIGURAÇÃO INICIAL (Baseado no seu 'tratamento.py') ---

# Garante que o pacote de stopwords do NLTK está baixado
try:
    stopwords.words('portuguese')
except LookupError:
    print("Baixando a lista de stopwords do NLTK (necessário apenas na primeira vez)...")
    nltk.download('stopwords')

# Cria o conjunto de stopwords em português para maior eficiência
stop_words_pt = set(stopwords.words('portuguese'))


# --- FUNÇÕES DE LIMPEZA (Exatamente como no seu arquivo 'tratamento.py') ---

def remover_urls_e_usernames(texto):
    """Remove padrões de URL e usernames do texto."""
    # Adicionado str(texto) para garantir que a função não falhe com dados não-textuais
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

# Nomes dos arquivos de entrada e saída
arquivo_bruto = r"C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_3\told_apenas_hate_2.0_bruto.csv"
arquivo_tratado_final = "told_apenas_hate_2.0_tratado.csv"

try:
    # 1. Carrega o dataset bruto que você gerou
    print(f"Carregando o arquivo '{arquivo_bruto}'...")
    df_bruto = pd.read_csv(arquivo_bruto)
    print("Arquivo carregado com sucesso.")
    print("\nAmostra dos dados brutos:")
    print(df_bruto.head().to_string())

    # Cria uma cópia para aplicar o tratamento
    df_tratado = df_bruto.copy()

    # 2. Aplica o fluxo de tratamento na coluna 'text'
    print("\n--- INICIANDO PROCESSO DE LIMPEZA ---")

    print("\n1. Removendo URLs e usernames...")
    df_tratado['text'] = df_tratado['text'].apply(remover_urls_e_usernames)

    print("2. Removendo pontuação...")
    df_tratado['text'] = df_tratado['text'].apply(remover_pontuacao)

    print("3. Removendo stopwords e convertendo para minúsculas...")
    df_tratado['text'] = df_tratado['text'].apply(remover_stopwords)

    print("4. Removendo espaços em excesso...")
    df_tratado['text'] = df_tratado['text'].apply(lambda x: " ".join(x.split()))

    # 3. Salva o resultado no novo arquivo CSV
    df_tratado.to_csv(arquivo_tratado_final, index=False, encoding='utf-8')

    print("\n--- SUCESSO! ---")
    print(f"Arquivo '{arquivo_tratado_final}' foi criado com o texto tratado.")
    print("Agora você pode avaliar a precisão do conteúdo limpo.")
    print("\nAmostra do arquivo final tratado:")
    print(df_tratado.head().to_string())


except FileNotFoundError:
    print(f"\nERRO: O arquivo '{arquivo_bruto}' não foi encontrado.")
    print("Por favor, certifique-se de que ele está na mesma pasta que este script.")