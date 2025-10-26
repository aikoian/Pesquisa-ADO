import pandas as pd

# 1. Carregar o dataset ToLD-BR
file_path_told = r'C:\Users\aiko\Documents\GitHub\Pesquisa-ADO\dataset_3\ToLD-BR.csv'
try:
    df_told = pd.read_csv(file_path_told)
except FileNotFoundError:
    print(f"Erro: Arquivo '{file_path_told}' não encontrado.")
    print("Por favor, coloque este script no mesmo diretório do 'ToLD-BR.csv'.")
    exit()

# 2. Definir a lógica [0,0] para o ToLD-BR
# Um caso [0,0] (neutro) é aquele onde *todas* as colunas de classificação
# são 0.0.
annotation_cols = ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']

# 3. Filtrar os casos [0,0]
# Somamos os valores de todas as colunas de anotação.
# Se a soma for 0.0, significa que todas são 0.0.
df_00_told = df_told[df_told[annotation_cols].sum(axis=1) == 0.0].copy()

# 4. Verificar a contagem
total_00_told = len(df_00_told)
n_amostras = 607

if total_00_told >= n_amostras:
    print(f"Total de casos 'limpos' [0,0] encontrados: {total_00_told}")
    
    # 5. Extrair 607 amostras aleatórias
    # random_state=42 garante que a amostra seja sempre a mesma
    df_00_sample = df_00_told.sample(n=n_amostras, random_state=42)
    
    # 6. Criar o DataFrame final no formato solicitado
    df_final = pd.DataFrame()
    df_final['text'] = df_00_sample['text']
    df_final['discurso_de_odio'] = 0
    df_final['ataque_individual'] = 0
    
    # 7. Salvar no arquivo CSV
    output_filename = 'amostra_607_casos_00_ToLD-BR.csv'
    
    # Salvar sem cabeçalho (header=False) e sem índice (index=False)
    # encoding='utf-8-sig' é recomendado para CSVs com acentos
    df_final.to_csv(output_filename, header=False, index=False, encoding='utf-8-sig')
    
    print(f"\nArquivo '{output_filename}' gerado com sucesso!")
    print(f"Total de linhas no arquivo: {len(df_final)}")

else:
    print(f"\nErro: Não foi possível gerar a amostra.")
    print(f"O ToLD-BR possui apenas {total_00_told} casos 'limpos', mas precisávamos de {n_amostras}.")