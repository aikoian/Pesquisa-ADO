import pandas as pd

# Defina o nome do arquivo de entrada que você já criou
arquivo_de_entrada = "D:\Pesquisa-ADO\Pesquisa\dataset_final.csv"

# Defina o nome do novo arquivo que será gerado
arquivo_de_saida = "hate_speech_only.csv"

try:
    # 1. Carrega o seu dataset completo e já processado
    print(f"Carregando o arquivo '{arquivo_de_entrada}'...")
    df_completo = pd.read_csv(arquivo_de_entrada)

    # 2. Filtra o DataFrame, mantendo apenas as linhas onde 'is_hate_speech' é igual a 1
    print("Filtrando para manter apenas os casos de discurso de ódio (is_hate_speech == 1)...")
    df_apenas_odio = df_completo[df_completo['hatespeech_comb'] == 1].copy()

    # 3. Verifica se foram encontrados exemplos
    if df_apenas_odio.empty:
        print("Nenhum exemplo de discurso de ódio (label 1) foi encontrado no arquivo.")
    else:
        # 4. Salva o resultado filtrado em um novo arquivo CSV
        df_apenas_odio.to_csv(arquivo_de_saida, index=False, encoding='utf-8')
        print("\n--- SUCESSO! ---")
        print(f"Arquivo '{arquivo_de_saida}' foi criado.")
        print(f"Ele contém {len(df_apenas_odio)} exemplos para sua avaliação.")
        
        # Mostra uma pequena amostra no terminal
        print("\nAmostra dos dados extraídos:")
        print(df_apenas_odio.head())

except FileNotFoundError:
    print(f"\nERRO: O arquivo '{arquivo_de_entrada}' não foi encontrado.")
    print("Por favor, certifique-se de que ele está na mesma pasta que este script.")