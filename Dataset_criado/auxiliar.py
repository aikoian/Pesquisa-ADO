

import pandas as pd
import os
from glob import glob

# Caminhos dos arquivos (ajuste conforme necessário)
arquivos = [
r"C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_4\olid_gerados_[0,0]\amostra_de_607_[0,0]_para_revisao.csv"
r"C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_4\olid_gerados_[0,1]\amostra_de_607_[0,1]_para_revisao.csv"
r"C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_4\olid_gerados_[1,1]\607_casos_[1,1]_gerados.csv"
r"C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_4\olid_gerados_[1.0]\casos_[1,0]_gerado.csv"
]

# Função auxiliar para padronizar colunas
def padronizar_colunas(df):
    colunas = df.columns.str.lower().str.strip()  # normaliza letras
    df.columns = colunas
    # tenta identificar colunas
    if 'text' not in df.columns:
        df.rename(columns={colunas[0]: 'text'}, inplace=True)
    # adiciona labels genéricas se faltarem
    if len(df.columns) == 2:
        df.columns = ['text', 'label_1']
        df['label_2'] = None
    elif len(df.columns) == 3:
        df.columns = ['text', 'label_1', 'label_2']
    return df

# Lista para armazenar dataframes
dfs = []

for arquivo in arquivos:
    if os.path.exists(arquivo):
        df = pd.read_csv(arquivo)
        df = padronizar_colunas(df)
        dfs.append(df)
        print(f"✅ Arquivo carregado: {arquivo} — {len(df)} linhas")
    else:
        print(f"⚠️ Arquivo não encontrado: {arquivo}")

# Combina todos os dataframes
if dfs:
    combinado = pd.concat(dfs, ignore_index=True)
    combinado = combinado.sample(frac=1, random_state=42).reset_index(drop=True)  # embaralha

    # Salva o resultado final
    combinado.to_csv("dataset_combinado.csv", index=False, encoding="utf-8-sig")

    print("\n✅ Dataset combinado salvo como 'dataset_combinado.csv'")
    print(f"📊 Total de linhas: {len(combinado)}")
    print("\nPrévia das 5 primeiras linhas:\n")
    print(combinado.head())
else:
    print("\n❌ Nenhum arquivo válido foi carregado.")
