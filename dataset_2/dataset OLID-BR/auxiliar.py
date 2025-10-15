import pandas as pd

# Nome do arquivo original que você enviou
arquivo_base = "D:\Pesquisa-ADO\dataset_2\dataset OLID-BR\olid_br_base.csv"

# Nome do novo arquivo, focado apenas nos casos positivos para revisão
arquivo_para_revisao = "casos_positivos_para_revisao.csv"

try:
    # 1. Carrega o dataset completo
    print(f"Carregando o arquivo base '{arquivo_base}'...")
    df = pd.read_csv(arquivo_base)

    # 2. Define as colunas de interesse para a filtragem
    colunas_de_odio = [
        'racism',
        'sexism',
        'xenophobia',
        'lgbtqphobia',
        'religious_intolerance'
    ]
    
    # Garante que as colunas de ódio sejam do tipo booleano (True/False) para a filtragem
    for col in colunas_de_odio:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # 3. FILTRAGEM DE LINHAS: Mantém apenas as linhas onde QUALQUER uma das colunas de ódio for True
    # O comando .any(axis=1) verifica, para cada linha, se há algum 'True' nas colunas especificadas.
    print("Filtrando para manter apenas os casos positivos...")
    df_filtrado = df[df[colunas_de_odio].any(axis=1)].copy()

    # 4. SELEÇÃO DE COLUNAS: Define as colunas que queremos no arquivo final
    colunas_finais = ['text'] + colunas_de_odio
    df_final_para_revisao = df_filtrado[colunas_finais]

    # 5. Salva o resultado em um novo arquivo CSV
    df_final_para_revisao.to_csv(arquivo_para_revisao, index=False, encoding='utf-8')

    print("\n--- SUCESSO! ---")
    print(f"Arquivo '{arquivo_para_revisao}' foi criado e está pronto para sua análise manual.")
    print(f"Ele contém {len(df_final_para_revisao)} casos onde pelo menos um tipo de ataque foi identificado.")
    print("\nAmostra do arquivo gerado:")
    print(df_final_para_revisao.head().to_string())


except FileNotFoundError:
    print(f"\nERRO: O arquivo '{arquivo_base}' não foi encontrado.")
    print("Por favor, certifique-se de que ele está na mesma pasta que este script.")
except KeyError:
    print("\nERRO: Uma ou mais colunas necessárias não foram encontradas no arquivo original.")
    print("Por favor, verifique se os nomes das colunas no script correspondem exatamente aos do seu CSV.")