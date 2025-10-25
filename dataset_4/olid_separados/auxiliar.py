import pandas as pd
import sys

# --- Configuração ---
# O número de amostras para igualar à sua classe de ódio a grupo [1,0]
# com base no arquivo casos_[1,0]_gerado.csv
N_AMOSTRAS = 607

# Arquivo de entrada (o grande)
ARQUIVO_DE_ENTRADA = r"C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_4\olid_separados\casos_ataque_individual.csv"

# Arquivo de saída (a amostra para você revisar)
ARQUIVO_DE_SAIDA = "amostra_ataque_individual_para_revisao.csv"

# Estado aleatório para garantir que a amostragem seja reprodutível
RANDOM_STATE = 42

try:
    # 1. Carregar o arquivo de ataque individual
    print(f"Carregando o arquivo completo '{ARQUIVO_DE_ENTRADA}'...")
    df = pd.read_csv(ARQUIVO_DE_ENTRADA)
    total_linhas_entrada = len(df)
    print(f"Arquivo carregado com sucesso. Total de {total_linhas_entrada} linhas.")

    # 2. Verificar se temos linhas suficientes
    if total_linhas_entrada == 0:
        print("ERRO: O arquivo de entrada está vazio. Não é possível gerar amostra.", file=sys.stderr)
        sys.exit()
    elif total_linhas_entrada < N_AMOSTRAS:
        print(f"AVISO: O arquivo de entrada tem apenas {total_linhas_entrada} linhas, que é menos do que as {N_AMOSTRAS} solicitadas.")
        print("O arquivo de amostra conterá todas as linhas disponíveis.")
        n_amostra_real = total_linhas_entrada
    else:
        n_amostra_real = N_AMOSTRAS
        print(f"Selecionando uma amostra aleatória de {n_amostra_real} linhas...")

    # 3. Selecionar a amostra aleatória
    # replace=False garante que não teremos linhas duplicadas na amostra
    df_amostra = df.sample(n=n_amostra_real, random_state=RANDOM_STATE, replace=False)

    # 4. Salvar a amostra em um novo arquivo CSV
    df_amostra.to_csv(ARQUIVO_DE_SAIDA, index=False, encoding='utf-8')

    print(f"\n--- SUCESSO! ---")
    print(f"Arquivo '{ARQUIVO_DE_SAIDA}' foi salvo na sua pasta.")
    print(f"Este arquivo contém {len(df_amostra)} linhas para sua curadoria manual.")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{ARQUIVO_DE_ENTRADA}' não foi encontrado.", file=sys.stderr)
    print("Por favor, certifique-se que o arquivo está na mesma pasta do script.")
except Exception as e:
    print(f"Um erro inesperado ocorreu: {e}", file=sys.stderr)