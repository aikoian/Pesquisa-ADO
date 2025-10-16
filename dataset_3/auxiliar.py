import pandas as pd

# --- CONFIGURAÇÃO ---
# Coloque aqui o nome do arquivo CSV completo do ToLD-Br
arquivo_base_toldbr = r"C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_3\ToLD-BR.csv" 

# Nome do arquivo que será gerado com os dados filtrados
arquivo_final_para_analise = "toldbr_casos_relevantes.csv"

# Colunas que queremos manter no arquivo final
colunas_de_interesse = [
    'text',
    'homophobia',
    'racism',
    'misogyny',
    'xenophobia'
]

# Colunas que usaremos para filtrar os casos de ódio explícito
colunas_de_odio = [
    'homophobia',
    'racism',
    'misogyny',
    'xenophobia'
]


# --- EXECUÇÃO DO SCRIPT ---
try:
    # 1. Carrega o dataset completo
    print(f"Carregando o arquivo '{arquivo_base_toldbr}'...")
    df = pd.read_csv(arquivo_base_toldbr)

    # 2. FILTRAGEM DE LINHAS: 
    # Mantém apenas as linhas onde QUALQUER uma das colunas de ódio seja igual a 2.0
    # Isso garante que estamos pegando apenas os casos de alta confiança de discurso de ódio.
    print("Filtrando para manter apenas os casos de ódio explícito (valor == 2.0)...")
    condicao_odio_explicito = (df[colunas_de_odio] == 1.0).any(axis=1)
    df_filtrado = df[condicao_odio_explicito]

    # 3. SELEÇÃO DE COLUNAS:
    # Cria o dataframe final apenas com as colunas que você quer visualizar.
    df_final = df_filtrado[colunas_de_interesse]

    # 4. Salva o resultado no novo arquivo CSV
    df_final.to_csv(arquivo_final_para_analise, index=False, encoding='utf-8')

    print("\n--- SUCESSO! ---")
    print(f"Arquivo '{arquivo_final_para_analise}' foi criado.")
    print(f"Ele contém {len(df_final)} casos de alta confiança para sua análise manual.")
    print("\nAmostra do arquivo gerado:")
    print(df_final.head().to_string())


except FileNotFoundError:
    print(f"\nERRO: O arquivo '{arquivo_base_toldbr}' não foi encontrado.")
    print("Por favor, renomeie o seu arquivo CSV do ToLD-Br para o nome definido na variável 'arquivo_base_toldbr' ou altere a variável no script.")
except KeyError:
    print("\nERRO: Uma ou mais colunas não foram encontradas no arquivo.")
    print("Por favor, verifique se os nomes das colunas no script correspondem exatamente aos do seu CSV.")