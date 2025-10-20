# Passo 1: Importar a biblioteca pandas
# Damos a ela o apelido 'pd', que é uma convenção comum e facilita a escrita do código.
import pandas as pd

# Passo 2: Carregar o arquivo CSV para a memória
# A função pd.read_csv() lê o seu arquivo e o transforma em uma estrutura de dados
# chamada DataFrame, que funciona como uma tabela ou planilha.
# Vamos armazenar nosso DataFrame na variável 'df'.
print("Carregando o arquivo 'dataset_multirotulo_para_revisao.csv'...")
try:
    df = pd.read_csv(r"C:\Users\Lenovo\Documents\GitHub\Pesquisa-ADO\dataset_4\dataset_multirotulo_para_revisao.csv")
    print("Arquivo carregado com sucesso!")
except FileNotFoundError:
    print("Erro: O arquivo 'dataset_multirotulo_para_revisao.csv' não foi encontrado.")
    print("Por favor, certifique-se de que o script e o arquivo CSV estão na mesma pasta.")
    exit() # Encerra o script se o arquivo não for encontrado

# --- Início da Filtragem ---

# Passo 3.1: Filtrar e salvar os casos de "Discurso de Ódio"
# Aqui, criamos uma condição: df['discurso_de_odio'] == 1
# Isso seleciona apenas as linhas onde o valor na coluna 'discurso_de_odio' é 1.
# O resultado é um novo DataFrame, que salvamos em 'df_odio'.
print("\nFiltrando casos de 'Discurso de Ódio'...")
df_odio = df[df['discurso_de_odio'] == 1]

# Agora, salvamos este novo DataFrame em um novo arquivo CSV.
# Usamos index=False para não salvar os números das linhas do DataFrame como uma coluna extra.
# Usamos encoding='utf-8' para garantir que caracteres especiais (acentos, ç) sejam salvos corretamente.
df_odio.to_csv("casos_discurso_de_odio.csv", index=False, encoding='utf-8')
print(f"-> Arquivo 'casos_discurso_de_odio.csv' criado com {len(df_odio)} linhas.")


# Passo 3.2: Filtrar e salvar os casos de "Ataque Individual"
# O processo é o mesmo, mas agora a condição é para a coluna 'ataque_individual'.
print("\nFiltrando casos de 'Ataque Individual'...")
df_ataque = df[df['ataque_individual'] == 1]
df_ataque.to_csv("casos_ataque_individual.csv", index=False, encoding='utf-8')
print(f"-> Arquivo 'casos_ataque_individual.csv' criado com {len(df_ataque)} linhas.")


# Passo 3.3: Filtrar e salvar os casos que não são nem ódio nem ataque
# Aqui, a condição é composta. Queremos linhas onde 'discurso_de_odio' é 0 E (&) 'ataque_individual' é 0.
# Em pandas, cada condição deve estar entre parênteses quando usamos o operador '&' (E).
print("\nFiltrando casos 'Nenhum'...")
df_nenhum = df[(df['discurso_de_odio'] == 0) & (df['ataque_individual'] == 0)]
df_nenhum.to_csv("casos_nenhum.csv", index=False, encoding='utf-8')
print(f"-> Arquivo 'casos_nenhum.csv' criado com {len(df_nenhum)} linhas.")

print("\nProcesso concluído com sucesso!")