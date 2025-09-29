import pandas as pd
import re
import spacy

# --- CARREGAMENTO DO MODELO SPACY ---
try:
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
    print("Modelo 'pt_core_news_sm' do spaCy carregado com sucesso.")
except OSError:
    print("Modelo 'pt_core_news_sm' não encontrado. Por favor, execute:")
    print("python -m spacy download pt_core_news_sm")
    exit()

def limpeza_final_para_treino(texto: str) -> str:
    """
    Executa uma limpeza final e otimizada em um texto, preparando-o para modelos de NLP.
    """
    if not isinstance(texto, str):
        return ""

    # Etapa 1: Converter para minúsculas (GARANTE CONSISTÊNCIA DE CASO)
    texto = texto.lower()

    # Etapa 2: Remover URLs, @menções e #hashtags
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'\@\w+|\#', '', texto)

    # --- AJUSTE CRÍTICO AQUI ---
    # Etapa 3: Remover caracteres especiais, mas MANTER ACENTOS E Ç.
    # Isso corrige o problema de 'saúde' -> 'sade'.
    texto = re.sub(r'[^a-zà-úç\s]', '', texto)

    # Etapa 4: Processar o texto com spaCy para lematização e remoção de stopwords
    doc = nlp(texto)

    tokens_limpos = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha and len(token.lemma_) > 1
    ]
    texto_limpo = ' '.join(tokens_limpos)
    
    # Etapa 5: Normalizar espaços em branco (etapa final de polimento)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    return texto_limpo

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    caminho_entrada = r'C:\Users\aikoi\OneDrive\Área de Trabalho\Arquivos\Estudos\Pesquisa ADO testes\dataset_original.csv'
    caminho_saida = 'dataset_pronto_para_treino.csv'

    try:
        # Ler APENAS as duas primeiras colunas do arquivo original
        print(f"Carregando as duas primeiras colunas de: {caminho_entrada}")
        df = pd.read_csv(caminho_entrada, header=None, usecols=[0, 1])

        # Nomear as colunas
        df.columns = ['text', 'hatespeech_comb']
        
        print("Dataset carregado. Amostra inicial:")
        print(df.head())

    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_entrada}' não foi encontrado.")
        exit()
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")
        exit()

    # Aplica a limpeza final, substituindo a coluna de texto original
    print("\nAplicando limpeza final no dataset...")
    df['text'] = df['text'].apply(limpeza_final_para_treino)
    
    # Etapa extra: Remover linhas que possam ter ficado vazias após a limpeza
    df.dropna(subset=['text'], inplace=True)
    df = df[df['text'] != '']
    
    print("Limpeza concluída.")

    # Exibe as 10 primeiras linhas do DataFrame final e pronto
    print("\nAmostra do resultado final, pronto para o treino:")
    print(df.head(10))

    # Salva o DataFrame final, que agora contém apenas as duas colunas tratadas
    df.to_csv(caminho_saida, index=False)
    print(f"\nDataset final salvo em '{caminho_saida}'.")
    print("O arquivo está pronto para ser usado no treinamento de modelos.")