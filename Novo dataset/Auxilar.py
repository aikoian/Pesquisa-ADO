import pandas as pd

# --- Passo 1: Carregar o dataset OLiD-BR ---
try:
    df_olid = pd.read_csv("train.csv")
    print("Dataset OLiD-BR (train.csv) carregado com sucesso!")
except FileNotFoundError as e:
    print(f"Erro: Arquivo train.csv não encontrado.\nDetalhe: {e}")
    exit()

# --- Passo 2: Criar a coluna 'label_odio' com a lógica precisa ---
# Primeiro, vamos criar uma cópia para não modificar o original
df_preciso = df_olid[['text', 'is_offensive', 'is_targeted']].copy()

# Definir as condições para cada classe
condicao_odio = (df_preciso['is_targeted'] == 'TIN')
condicao_nao_odio = (df_preciso['is_offensive'] == 'NOT')

# Aplicar as condições para criar a nova coluna
df_preciso.loc[condicao_odio, 'label_odio'] = 1
df_preciso.loc[condicao_nao_odio, 'label_odio'] = 0

# --- Passo 3: Limpar o dataset ---
# Remover as linhas que não se encaixam em nenhuma das duas categorias (os xingamentos genéricos)
df_preciso.dropna(subset=['label_odio'], inplace=True)

# Converter a coluna para inteiro
df_preciso['label_odio'] = df_preciso['label_odio'].astype(int)

# Selecionar apenas as colunas finais
df_preciso = df_preciso[['text', 'label_odio']]
df_preciso.rename(columns={'text': 'texto'}, inplace=True)
print("Dataset filtrado para conter apenas 'Discurso de Ódio Direcionado' (1) e 'Não Ofensivo' (0).")


# --- Passo 4: Balancear e Salvar ---
# Separar por classe para fazer o balanceamento
df_odio = df_preciso[df_preciso['label_odio'] == 1]
df_nao_odio = df_preciso[df_preciso['label_odio'] == 0]

print(f"\n--- Análise Antes do Balanceamento ---")
print(f"Total de 'Não Discurso de Ódio' (0): {len(df_nao_odio)}")
print(f"Total de 'Discurso de Ódio' (1): {len(df_odio)}")

# Fazer subamostragem da classe majoritária (que provavelmente será 'Não Discurso de Ódio')
n_samples = len(df_odio) # O número de amostras será o da classe minoritária
df_nao_odio_balanceado = df_nao_odio.sample(n=n_samples, random_state=42)

# Juntar as duas classes balanceadas
df_final = pd.concat([df_odio, df_nao_odio_balanceado])
print(f"\nBalanceando: Reduzindo a classe 'Não Discurso de Ódio' para {n_samples} amostras.")


# Embaralhar e salvar
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
df_final.to_csv('dataset_focado_em_discurso_de_odio.csv', index=False)

print("\n--- Processo Finalizado ---")
print(f"Total de amostras no novo dataset: {len(df_final)}")
print(f"Distribuição final das classes:\n{df_final['label_odio'].value_counts()}")
print("\nArquivo 'dataset_focado_em_discurso_de_odio.csv' foi criado com sucesso!")