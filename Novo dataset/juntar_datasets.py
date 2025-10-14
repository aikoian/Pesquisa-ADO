"""
links importantes:

link do dataset OLID-BR: https://www.kaggle.com/datasets/dougtrajano/olidbr?resource=download
link do dataset HateBR : https://github.com/franciellevargas/HateBR/blob/main/dataset/HateBR.csv

"""\

#! deu aguia total '-'
def linha():
    print("-="*115)
    print()

# minha ideia era fazer a junção de dois datasets, porém me encontrei numa confunsão absurda, mas acabei chegando em uma solução melhor

import pandas as pd

try:
    df_olid = pd.read_csv(r"D:\Pesquisa-ADO\Novo dataset\train.csv")
    print("O dataset foi carregado com sucesso.")
except FileNotFoundError:
    print("O arquivo não pode ser carregado, verifique o caminho do arquivo, ou anexe um (r) antes da string.")
    exit()

# clonar o meu df pra não alterar o arquivo original e criar a coluna de 'label-odio'

# DEFINIÇÃO RESTRITA E FINAL DE DISCURSO DE ÓDIO
colunas_odio_restrito = [
    'racism',
    'sexism',
    'lgbtqphobia',
    'xenophobia',
    'religious_intolerance'
]

# Copiar apenas as colunas necessárias
df_preciso = df_olid[['text', 'is_offensive'] + colunas_odio_restrito].copy()

# A condição de ódio é VERDADEIRA se QUALQUER uma das colunas RESTRITAS for True
condicao_odio = df_preciso[colunas_odio_restrito].any(axis=1)
condicao_nao_odio = (df_preciso['is_offensive'] == 'NOT')

# criar uma nova coluna
df_preciso.loc[condicao_odio, 'label_odio'] = 1
df_preciso.loc[condicao_nao_odio, 'label_odio'] = 0

#remover as linhas que não vão ter utilidade
df_preciso.dropna(subset=['label_odio'], inplace=True)

# converter a coluna para INT
df_preciso['label_odio'] = df_preciso['label_odio'].astype(int)

# selecionar as colunas finais
df_preciso = df_preciso[['text', 'label_odio']]
df_preciso.rename(columns={'text': 'texto'}, inplace=True)
print("Dataset filtrado com sucesso usando a definição MAIS RESTRITA de discurso de ódio.")


# agora vou balancear e salvar em um novo arquivo
df_odio = df_preciso[df_preciso['label_odio'] == 1]
df_nao_odio = df_preciso[df_preciso['label_odio'] == 0]
linha()
print(f"""
todal de 'Não discurso de ódio' = {len(df_nao_odio)}

total de 'Discurso de ódio  = {len(df_odio)}

""")

# Lógica de balanceamento dinâmica: sempre reduz a classe maior para o tamanho da menor.
if len(df_nao_odio) > len(df_odio):
    classe_maior = df_nao_odio
    classe_menor = df_odio
    df_maior_balanceado = classe_maior.sample(n=len(classe_menor), random_state=42)
    df_final = pd.concat([classe_menor, df_maior_balanceado])
    print(f"\nBalanceando: Reduzindo a classe 'Não Discurso de Ódio' para {len(classe_menor)} amostras.")
else:
    classe_maior = df_odio
    classe_menor = df_nao_odio
    df_maior_balanceado = classe_maior.sample(n=len(classe_menor), random_state=42)
    df_final = pd.concat([classe_menor, df_maior_balanceado])
    print(f"\nBalanceando: Reduzindo a classe 'Discurso de Ódio' para {len(classe_menor)} amostras.")


# Embaralhar e salvar
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
df_final.to_csv('dataset_definitivo.csv', index=False)

print("\n--- Processo Finalizado ---")
print(f"Total de amostras no novo dataset: {len(df_final)}")
print(f"Distribuição final das classes:\n{df_final['label_odio'].value_counts()}")
print("\nArquivo 'dataset_definitivo.csv' foi criado com sucesso!")