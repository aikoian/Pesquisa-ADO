import pandas as pd

# Nome do arquivo original
arquivo_base = "D:\Pesquisa-ADO\dataset_2\dataset OLID-BR\olid_br_base.csv"

# Nome do novo arquivo, agora com a filtragem mais precisa
arquivo_para_revisao = "casos_positivos_GRP_para_revisao.csv"

try:
    # 1. Carrega o dataset completo
    print(f"Carregando o arquivo base '{arquivo_base}'...")
    df = pd.read_csv(arquivo_base)

    # 2. Define as colunas de interesse para a filtragem de tipo de ódio
    colunas_de_odio = [
        'racism',
        'sexism',
        'xenophobia',
        'lgbtqphobia',
        'religious_intolerance'
    ]
    
    # Garante que as colunas de ódio sejam do tipo booleano (True/False)
    for col in colunas_de_odio:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # 3. FILTRAGEM DUPLA: Aplica as duas condições com o operador AND (&)
    # Condição A: O ataque deve ser a um grupo (GRP)
    condicao_grupo = (df['targeted_type'] == 'GRP')
    # Condição B: Pelo menos um tipo de ódio deve ser True
    condicao_tipo_odio = df[colunas_de_odio].any(axis=1)

    print("Filtrando para manter apenas os casos que são ataques a grupos (GRP) e possuem um tipo de ódio específico...")
    df_filtrado = df[condicao_grupo & condicao_tipo_odio].copy()

    # 4. SELEÇÃO DE COLUNAS: Define as colunas que queremos no arquivo final
    colunas_finais = ['text'] + colunas_de_odio
    df_final_para_revisao = df_filtrado[colunas_finais]

    # 5. Salva o resultado no novo arquivo CSV
    df_final_para_revisao.to_csv(arquivo_para_revisao, index=False, encoding='utf-8')

    print("\n--- SUCESSO! ---")
    print(f"Arquivo '{arquivo_para_revisao}' foi criado e está pronto para sua análise manual.")
    print(f"Ele contém {len(df_final_para_revisao)} casos que são ataques diretos a grupos com um tipo de ódio identificado.")
    print("\nAmostra do arquivo gerado:")
    print(df_final_para_revisao.head().to_string())


except FileNotFoundError:
    print(f"\nERRO: O arquivo '{arquivo_base}' não foi encontrado.")
    print("Por favor, certifique-se de que ele está na mesma pasta que este script.")
except KeyError as e:
    print(f"\nERRO: A coluna {e} não foi encontrada no arquivo original.")
    print("Por favor, verifique se os nomes das colunas no script correspondem exatamente aos do seu CSV.")