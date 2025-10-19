import pandas as pd

# --- Nomes dos arquivos de entrada ---
# (Certifique-se que esses arquivos estão na mesma pasta que o script)
arquivo_textos_e_alvos = "D:\Pesquisa-ADO\dataset_4\olid_br_test.csv"
arquivo_metadados_odio = "D:\Pesquisa-ADO\dataset_4\olid_br_train.csv"

# --- Nome do arquivo de saída que vamos criar ---
arquivo_base_completo = "olid_br_base.csv"

print("Iniciando o processo de junção dos arquivos...")

try:
    # 1. Carregar os dois arquivos CSV
    print(f"Carregando '{arquivo_textos_e_alvos}' (textos e alvos)...")
    df_train = pd.read_csv(arquivo_textos_e_alvos)
    
    print(f"Carregando '{arquivo_metadados_odio}' (metadados de ódio)...")
    df_metadata = pd.read_csv(arquivo_metadados_odio)

    # 2. Mesclar (merge) os dois DataFrames usando a coluna 'id' como chave
    print("Mesclando os dois arquivos usando a coluna 'id'...")
    # 'how="left"' garante que manteremos todos os textos, mesmo se não tiverem metadados de ódio
    df_merged = pd.merge(df_train, df_metadata, on="id", how="left")

    # 3. CRUCIAL: Remover duplicatas que surgiram do merge
    # Como descobrimos, 'train_metadata.csv' pode ter várias linhas por 'id'.
    # Vamos manter apenas a primeira ocorrência de cada 'id'.
    print(f"Removendo duplicatas... (Antes: {len(df_merged)} linhas)")
    df_merged.drop_duplicates(subset=['id'], keep='first', inplace=True)
    print(f"Tamanho final após remover duplicatas: {len(df_merged)} linhas")

    # 4. Salvar o arquivo base completo
    df_merged.to_csv(arquivo_base_completo, index=False, encoding='utf-8')
    
    print("\n--- SUCESSO! ---")
    print(f"Arquivo '{arquivo_base_completo}' foi criado com sucesso.")
    print("Este arquivo agora contém todas as colunas necessárias (texto, alvo e tipos de ódio).")

except FileNotFoundError as e:
    print(f"\nERRO: Arquivo não encontrado!")
    print(f"Não consegui encontrar o arquivo: {e.filename}")
    print("Por favor, verifique se os nomes dos arquivos CSV estão corretos e na mesma pasta do script.")
except Exception as e:
    print(f"\nUm erro inesperado ocorreu: {e}")