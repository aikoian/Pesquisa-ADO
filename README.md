# Análise de Discurso de ódio.
Desenvolvimento da pesquisa sobre discurso de ódio.

---- Limpeza do Dataset de Discurso de Ódio ----

O objetivo deste script é preparar e limpar o dataset para que ele possa ser usado em análises posteriores.

---- O que o script faz? ----
1. Carrega o dataset original escolhido
- Lemos um arquivo CSV que contém os textos e as anotações de discursos de ódio..
- Mantemos apenas duas colunas principais:
 - text (texto original).
 - hatespeech_comb (a classificação - se é reconhecido como discurso de ódio ou não).

2. Remove padrões de URL e usernames do texto.
 - elimina todo texto associado a (http, https, www) e @.
 - Isso deixa os textos mais limpos de ánalise.

3. Remove stopwords (palavras comuns que não ajudam na classificação)
 - Usamos as stopwords do NLTK em portugues (palavras como "de", "a","o", "para").
 - Isso ajuda a focar apenas nas palavras que realmente carregam significado.

4. Remove pontuação e símbolos desnecessários
 - Tiramos vírgulas, pontos, pontos de exclamação, interrogações etc.
 - Isso deixa os textos mais limpos de ánalise.

5. Normaliza os espaços
 - Caso existam espaços duplos ou em excesso, eles são corrigidos.

6. Exporta o dataset tratado
 - O arquivo final é salvo como dataset_tratado_final.csv, pronto para ser usado em modelos.

---- Como usar ----
1. Instale as bibliotecas necessárias:
 - pip install pandas nltk
2. Baixe as stopwords do NLTK (apenas uma vez):
 - import nltk
   nltk.download('stopwords')
3. Rode o script no seu ambiente Python.
4. O arquivo limpo (dataset_tratado_final.csv) será gerado na pasta do projeto.

----Estrutura esperada ----
1. Entrada( Dataset.csv)

| text                    | hatespeech_comb  |
| ----------------------- | ---------------- |
| "esse cara é um idiota" | 1                |
| "bom dia a todos"       | 0                |

2. Saída( dataset_tratado_final.csv)

| text            | hatespeech_comb  |
| --------------- | ---------------- |
| "cara idiota"   | 1                |
| "bom dia todos" | 0                |

---- Objetivo ----
1. Esse tratamento é um pré-processamento:
- Deixa os dados mais organizados.
- Evita "ruído" durante o treinamento de modelos.
- Aumenta a qualidade da análise de discurso de ódio.

