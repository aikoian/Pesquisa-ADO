# ==============================================================================
# SCRIPT DE TREINAMENTO E AVALIAÇÃO: NAIVE BAYES MULTINOMIAL
# Projeto: Detecção de Discurso de Ódio e Linguagem Ofensiva (Pesquisa-ADO)
# ==============================================================================

# Imports das bibliotecas fundamentais de manipulação de sistema e dados
import os           # Para verificação de existência de pastas e arquivos no sistema
import joblib       # Para salvar e carregar os modelos treinados em disco (.pkl)
import pandas as pd # Para leitura, manipulação e conversão das estruturas de dados (CSV/DataFrames)
import numpy as np  # Para operações matriciais e numéricas eficientes

# Imports do Scikit-Learn para Processamento de Linguagem Natural e Machine Learning
from sklearn.model_selection import train_test_split  # Para divisão dos dados em Treino e Teste
from sklearn.feature_extraction.text import TfidfVectorizer # Para conversão do texto em matriz numérica TF-IDF
from sklearn.naive_bayes import MultinomialNB          # Algoritmo Naive Bayes focado em contagens/frequências textuais
from sklearn.multioutput import MultiOutputClassifier   # Wrapper para permitir classificação multi-rótulo simultânea

# Imports de métricas para avaliação detalhada da performance dos modelos
from sklearn.metrics import (
    classification_report,      # Gera Precision, Recall e F1-Score por classe
    confusion_matrix,           # Gera a matriz de confusão (Previsto vs Real)
    f1_score,                   # Métrica F1-Score (média harmônica de precisão e revocação)
    accuracy_score,             # Acurácia exata da predição
    hamming_loss                 # Perda por rótulo incorreto (métrica padrão para multi-rótulo)
)


def carregar_e_preparar_dados(caminho_csv):
    """
    Função responsável por ler o arquivo CSV e estruturar as variáveis de treino.
    Converte as combinações binárias [ataque_grupo, ataque_individual] em rótulos de 0 a 3.
    """
    # Verifica se o arquivo CSV realmente existe no caminho especificado
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {caminho_csv}")

    # Carrega o CSV para a memória utilizando o Pandas
    df = pd.read_csv(caminho_csv)
    
    # Define as colunas que obrigatoriamente precisam existir na base de dados
    colunas_necessarias = {'texto', 'ataque_grupo', 'ataque_individual'}
    if not colunas_necessarias.issubset(df.columns):
        raise ValueError(f"O CSV deve conter as colunas: {colunas_necessarias}")

    # Remove eventuais linhas onde o texto esteja vazio/nulo e reseta os índices da tabela
    df = df.dropna(subset=['texto']).reset_index(drop=True)

    # Mapeamento da Matriz Taxonômica de 4 Classes (Multi-Classe):
    # Classe 0: Neutro               -> [ataque_grupo=0, ataque_individual=0]
    # Classe 1: Ataque Individual    -> [ataque_grupo=0, ataque_individual=1]
    # Classe 2: Discurso de Ódio     -> [ataque_grupo=1, ataque_individual=0]
    # Classe 3: Misto                -> [ataque_grupo=1, ataque_individual=1]
    def definir_classe_num(linha):
        grupo = linha['ataque_grupo']
        indiv = linha['ataque_individual']
        
        if grupo == 0 and indiv == 0:
            return 0  # Neutro [0,0]
        elif grupo == 0 and indiv == 1:
            return 1  # Ataque Individual [0,1]
        elif grupo == 1 and indiv == 0:
            return 2  # Discurso de Ódio [1,0]
        elif grupo == 1 and indiv == 1:
            return 3  # Misto [1,1]
        return 0

    # Aplica a função de mapeamento linha por linha criando uma nova coluna chamada 'classe_4cat'
    df['classe_4cat'] = df.apply(definir_classe_num, axis=1)

    # Separação dos Vetores de Entrada (X) e dos Rótulos Alvo (Y)
    X = df['texto']                                                      # O texto puro da frase
    Y_multirotulo = df[['ataque_grupo', 'ataque_individual']].values    # Matriz com 2 colunas binárias [G, I]
    y_multiclasse = df['classe_4cat'].values                             # Vetor simples com valores 0, 1, 2 ou 3

    return X, Y_multirotulo, y_multiclasse


def treinar_e_avaliar_pipeline(caminho_csv):
    """
    Função principal de treinamento: lê os dados, vetoriza, treina dois modelos Naive Bayes
    (Multi-Classe e Multi-Rótulo) e exibe o relatório analítico no console.
    """
    print(f"-> Carregando dataset a partir de: {caminho_csv}\n")
    X, Y_multi, y_cat = carregar_e_preparar_dados(caminho_csv)

    # Divisão de Dados em Treino (80%) e Teste (20%)
    # stratify=y_cat garante que a proporção exata de cada uma das 4 classes seja mantida no treino e teste
    X_treino, X_teste, y_cat_treino, y_cat_teste, Y_multi_treino, Y_multi_teste = train_test_split(
        X, y_cat, Y_multi,
        test_size=0.20,      # 20% das amostras reservadas para teste
        random_state=42,     # Semente fixa para garantir reprodutibilidade nos experimentos
        stratify=y_cat       # Mantém o balanço original das classes em ambas as divisões
    )

    print("==========================================================")
    print(" 1. VETORIZAÇÃO DE TEXTO (TF-IDF)")
    print("==========================================================")
    
    # Configuração do Vetorizador TF-IDF (Term Frequency - Inverse Document Frequency)
    # Convertemos o texto para uma matriz de relevância numérica de termos.
    vetorizador = TfidfVectorizer(
        ngram_range=(1, 2), # Considera palavras isoladas (unigramas) e pares de palavras (bigramas)
        sublinear_tf=True,  # Aplica escala logarítmica (1 + log(tf)) para amenizar o peso de palavras repetitivas
        min_df=2,           # Descarta termos que apareçam em menos de 2 documentos (reduz ruído)
        max_df=0.90         # Descarta termos que apareçam em mais de 90% dos documentos (palavras genéricas demais)
    )

    # Ajusta o dicionário com o texto de treino e transforma o treino em matriz numérica
    X_treino_tfidf = vetorizador.fit_transform(X_treino)
    
    # Apenas transforma o texto de teste usando o dicionário já aprendido no treino (evita vazamento de dados)
    X_teste_tfidf = vetorizador.transform(X_teste)

    print(f"Tamanho do Vocabulário aprendido pelo TF-IDF: {len(vetorizador.get_feature_names_out())} termos.\n")

    # =========================================================================
    # MODELO 1: CLASSFICADOR NAIVE BAYES MULTI-CLASSE (4 CATEGORIAS)
    # =========================================================================
    print("==========================================================")
    print(" 2. MODELO MULTI-CLASSE DIRETAS (4 CATEGORIAS: 0, 1, 2, 3)")
    print("==========================================================")

    # Instancia o Naive Bayes Multinomial com Suavização de Laplace (alpha=0.5)
    modelo_multiclasse = MultinomialNB(alpha=0.5)
    
    # Treina o modelo associando a matriz TF-IDF de treino com as categorias numéricas (0, 1, 2, 3)
    modelo_multiclasse.fit(X_treino_tfidf, y_cat_treino)

    # Realiza a predição para os dados de teste que o modelo nunca viu
    y_pred_cat = modelo_multiclasse.predict(X_teste_tfidf)

    # Nomes legíveis das categorias para impressão no relatório
    nomes_rotulos = ['Neutro [0,0]', 'Ataque Ind. [0,1]', 'Discurso Ódio [1,0]', 'Misto [1,1]']

    # Relatório de Métricas Detalhadas (Precision, Recall, F1-Score por classe)
    print("\n--- Relatório de Desempenho (Multi-Classe) ---")
    print(classification_report(y_cat_teste, y_pred_cat, target_names=nomes_rotulos, digits=4))

    # Matriz de Confusão para identificar onde o modelo está errando/confundindo as classes
    print("--- Matriz de Confusão ---")
    matriz_cm = confusion_matrix(y_cat_teste, y_pred_cat)
    matriz_df = pd.DataFrame(matriz_cm, index=nomes_rotulos, columns=nomes_rotulos)
    print(matriz_df)
    print("\n")

    # =========================================================================
    # MODELO 2: CLASSIFICADOR NAIVE BAYES MULTI-RÓTULO (BINARY RELEVANCE)
    # =========================================================================
    print("==========================================================")
    print(" 3. MODELO MULTI-RÓTULO (CLASSIFICAÇÃO BINÁRIA INDEPENDENTE)")
    print("==========================================================")

    # O MultiOutputClassifier cria 2 instâncias do Naive Bayes em paralelo:
    # Um modelo dedicado unicamente a prever 'ataque_grupo' (0 ou 1)
    # Um modelo dedicado unicamente a prever 'ataque_individual' (0 ou 1)
    modelo_multirotulo = MultiOutputClassifier(MultinomialNB(alpha=0.5))
    
    # Treina os dois modelos binários simultaneamente
    modelo_multirotulo.fit(X_treino_tfidf, Y_multi_treino)

    # Realiza a predição da matriz de teste com 2 colunas
    Y_pred_multi = modelo_multirotulo.predict(X_teste_tfidf)

    # Cálculo de métricas específicas para problemas multi-rótulo
    perda_hamming = hamming_loss(Y_multi_teste, Y_pred_multi)            # Fração de rótulos incorretos
    acuracia_exata = accuracy_score(Y_multi_teste, Y_pred_multi)        # Porcentagem de acerto de AMBOS os rótulos ao mesmo tempo
    f1_macro = f1_score(Y_multi_teste, Y_pred_multi, average='macro')   # F1-Score considerando o peso igual das classes

    print(f"Hamming Loss (quanto menor, melhor): {perda_hamming:.4f}")
    print(f"Acurácia Exata do Par (Subset Accuracy): {acuracia_exata * 100:.2f}%")
    print(f"F1-Score Macro: {f1_macro:.4f}\n")

    # =========================================================================
    # SALVAMENTO DOS MODELOS E ARTEFATOS
    # =========================================================================
    pasta_destino = "modelos_salvos"
    os.makedirs(pasta_destino, exist_ok=True) # Cria a pasta se ela não existir
    
    # Salva o vetorizador e os dois modelos treinados em arquivos compactos .pkl
    joblib.dump(vetorizador, os.path.join(pasta_destino, "vetorizador_tfidf.pkl"))
    joblib.dump(modelo_multiclasse, os.path.join(pasta_destino, "naive_bayes_multiclasse.pkl"))
    joblib.dump(modelo_multirotulo, os.path.join(pasta_destino, "naive_bayes_multirotulo.pkl"))
    
    print(f"-> Modelos e vetorizador salvos com sucesso na pasta '{pasta_destino}/'!")

    return vetorizador, modelo_multiclasse


def testar_frase_individual(texto, vetorizador, modelo_multiclasse):
    """
    Função auxiliar para você testar frases avulsas diretamente no terminal.
    """
    mapa_categorias = {
        0: "Neutro [0,0]",
        1: "Ataque Individual [0,1]",
        2: "Discurso de Ódio [1,0]",
        3: "Misto [1,1]"
    }
    
    # Transforma o texto simples recebido no formato matricial TF-IDF
    vetor_texto = vetorizador.transform([texto])
    
    # Realiza a predição da classe (0, 1, 2 ou 3)
    classe_prevista = modelo_multiclasse.predict(vetor_texto)[0]
    
    # Extrai as probabilidades estimadas pelo Naive Bayes para cada uma das 4 classes
    probabilidades = modelo_multiclasse.predict_proba(vetor_texto)[0]
    
    print(f"\nEntrada: \"{texto}\"")
    print(f"Predição do Modelo: {mapa_categorias[classe_prevista]}")
    print("Probabilidades por Classe:")
    for idx, prob in enumerate(probabilidades):
        print(f"  - {mapa_categorias[idx]}: {prob * 100:.2f}%")


# ==============================================================================
# BLOCO DE EXECUÇÃO PRINCIPAL (MAIN)
# ==============================================================================
if __name__ == "__main__":
    
    # Lista de caminhos prováveis onde seu CSV pode estar guardado na árvore de diretórios do projeto
    caminhos_possiveis = [
        "Dataset_criado/dataset_gerado/dataset_formatado_tratado.csv",
        "Dataset_criado/dataset_gerado/dataset_formatado_bruto.csv",
        "Dataset_criado/dataset_consolidado.csv",
        "dataset_consolidado.csv"
    ]
    
    caminho_final = None
    for caminho in caminhos_possiveis:
        if os.path.exists(caminho):
            caminho_final = caminho
            break

    if caminho_final is None:
        print("[ERRO] Nenhum arquivo CSV do dataset foi encontrado nas pastas do projeto.")
        print("Por favor, verifique se o arquivo .csv está na pasta 'Dataset_criado/dataset_gerado/'.")
    else:
        # 1. Executa o ciclo de treino e avaliação completa
        vetorizador_treinado, modelo_treinado = treinar_e_avaliar_pipeline(caminho_final)

        # 2. Testes de inferência em tempo real para validação rápida no terminal
        print("\n" + "=" * 60)
        print(" 4. TESTES PRÁTICOS DE INFERÊNCIA EM FRASES")
        print("=" * 60)

        testar_frase_individual(
            "Acho que deveríamos analisar os dados com cuidado antes da reunião.", 
            vetorizador_treinado, modelo_treinado
        )
        testar_frase_individual(
            "Você é um idiota completo e não sabe fazer nada direito.", 
            vetorizador_treinado, modelo_treinado
        )
        testar_frase_individual(
            "Todos esses policiais são porcos corruptos.", 
            vetorizador_treinado, modelo_treinado
        )
        testar_frase_individual(
            "Seu viadinho de merda, cale a sua boca.", 
            vetorizador_treinado, modelo_treinado
        )