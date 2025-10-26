
import pandas as pd
import re # Para lidar com as quebras de linha

# Nome do arquivo de entrada (o que você me enviou)
input_file = r'C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_4\olid_separados\amostra_607_casos_[0,0]_.csv'

# Nome do arquivo de saída (o arquivo corrigido)
output_file = 'amostra_607_casos_00_formatada.csv'

try:
    # Carregar o arquivo CSV. 
    # header=None porque ele não tem cabeçalho.
    # Damos nomes às colunas para facilitar.
    df = pd.read_csv(input_file, header=None, names=['text', 'discurso_de_odio', 'ataque_individual'])

    # Garantir que a coluna de texto seja tratada como string
    df['text'] = df['text'].astype(str)

    # --- Ação de Limpeza ---
    # 1. Substituir todas as formas de quebra de linha (\r\n, \n, \r) 
    #    por um único espaço em branco.
    #    regex=True permite encontrar todos os tipos de quebra de linha.
    df['text'] = df['text'].str.replace(r'\r\n|\n|\r', ' ', regex=True)
    
    # 2. Remover quaisquer espaços em branco extras no início ou no fim
    #    que possam ter resultado da substituição.
    df['text'] = df['text'].str.strip()

    # Salvar o DataFrame limpo em um novo arquivo CSV
    # header=False e index=False para manter o formato "texto",0,0
    # encoding='utf-8-sig' ajuda a evitar problemas com acentos.
    df.to_csv(output_file, header=False, index=False, encoding='utf-8-sig')

    print(f"Script executado com sucesso!")
    print(f"Arquivo de entrada: '{input_file}'")
    print(f"Arquivo formatado gerado: '{output_file}'")

except FileNotFoundError:
    print(f"Erro: Arquivo de entrada '{input_file}' não encontrado.")
    print("Por favor, certifique-se de que este script está no mesmo diretório que o arquivo CSV.")
except Exception as e:
    print(f"Ocorreu um erro durante o processamento: {e}")