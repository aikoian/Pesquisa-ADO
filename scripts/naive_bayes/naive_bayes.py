# ==============================================================================
# SCRIPT DE TREINAMENTO OTIMIZADO: NAIVE BAYES (MÁXIMO DESEMPENHO PREDITIVO)
# Projeto: Detecção de Discurso de Ódio e Linguagem Ofensiva (Pesquisa-ADO)
# ==============================================================================

import os
import joblib
import pandas as pd
import numpy as np

# Imports para divisão e pré-processamento avançado
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Substituição para ComplementNB (variante superior do Naive Bayes para texto)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.multioutput import MultiOutputClassifier

# Métricas de avaliação
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    hamming_loss
)


def carregar_e_preparar_dados(caminho_csv):
    """
    Carrega o CSV, padroniza as colunas e mapeia a taxonomia de 4 classes.
    """
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo não encontrado em: {caminho_csv}")

    df = pd.read_csv(caminho_csv)

    # Mapeamento dinâmico para garantir compatibilidade com diferentes versões do CSV
    mapeamento_colunas = {
        'text': 'texto',
        'comentario': 'texto',
        'texto_tratado': 'texto',
        'text_cleaned': 'texto',
        'discurso_de_odio': 'ataque_grupo',
        'hate_speech': 'ataque_grupo',
        'grupo': 'ataque_grupo',
        'individual': 'ataque_individual'
    }
    
    df = df.rename(columns=mapeamento_colunas)

    colunas_necessarias = {'texto', 'ataque_grupo', 'ataque_individual'}
    if not colunas_necessarias.issubset(df.columns):
        print("\n[ERRO DE COLUNAS] As colunas encontradas no CSV foram:", df.columns.tolist())
        raise ValueError(f"O CSV precisa ter as colunas {colunas_necessarias}.")

    # Limpeza básica de nulos
    df = df.dropna(subset=['texto']).reset_index(drop=True)

    # Mapeamento da Matriz Taxonômica de 4 Classes (0 a 3)
    def definir_classe_num(linha):
        grupo = int(linha['ataque_grupo'])
        indiv = int(linha['ataque_individual'])
        
        if grupo == 0 and indiv == 0:
            return 0  # Neutro [0,0]
        elif grupo == 0 and indiv == 1:
            return 1  # Ataque Individual [0,1]
        elif grupo == 1 and indiv == 0:
            return 2  # Discurso de Ódio [1,0]
        elif grupo == 1 and indiv == 1:
            return 3  # Misto [1,1]
        return 0

    df['classe_4cat'] = df.apply(definir_classe_num, axis=1)

    X = df['texto'].astype(str)
    Y_multirotulo = df[['ataque_grupo', 'ataque_individual']].values
    y_multiclasse = df['classe_4cat'].values

    return X, Y_multirotulo, y_multiclasse


def criar_vetorizador_hibrido():
    """
    Constrói um vetorizador composto (FeatureUnion) combinando:
    1. N-gramas de palavras (1 a 3 palavras)
    2. N-gramas de caracteres internamente nas palavras (3 a 5 caracteres)
    """
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),      # Unigramas, Bigramas e Trigramas de palavras
        sublinear_tf=True,       # Escala logarítmica de frequência
        min_df=1,                # Captura mesmo palavras raras
        strip_accents='unicode', # Remove acentos para normalização
        lowercase=True
    )

    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),      # Subpalavras de 3 a 5 caracteres
        sublinear_tf=True,
        min_df=2,                # Exige pelo menos 2 ocorrências para subpalavras
        strip_accents='unicode',
        lowercase=True
    )

    # Unifica as duas estratégias em uma única matriz de atributos
    vetorizador_hibrido = FeatureUnion([
        ('palavras', word_vectorizer),
        ('caracteres', char_vectorizer)
    ])

    return vetorizador_hibrido


def treinar_e_avaliar_otimizado(caminho_csv):
    """
    Executa o pipeline otimizado de treinamento e avaliação.
    """
    print(f"-> Carregando dataset a partir de: {caminho_csv}\n")
    X, Y_multi, y_cat = carregar_e_preparar_dados(caminho_csv)

    # Divisão de Dados (80% Treino / 20% Teste) Estratificada
    X_treino, X_teste, y_cat_treino, y_cat_teste, Y_multi_treino, Y_multi_teste = train_test_split(
        X, y_cat, Y_multi,
        test_size=0.20,
        random_state=42,
        stratify=y_cat
    )

    print("==========================================================")
    print(" 1. VETORIZAÇÃO HÍBRIDA DE TEXTO (WORD + CHAR N-GRAMS)")
    print("==========================================================")
    
    vetorizador = criar_vetorizador_hibrido()
    X_treino_tfidf = vetorizador.fit_transform(X_treino)
    X_teste_tfidf = vetorizador.transform(X_teste)

    # Cálculo do total de features extraídas
    num_features = X_treino_tfidf.shape[1]
    print(f"Total de Atributos Extraídos no Vocabulário Híbrido: {num_features} dimensões.\n")

    # =========================================================================
    # MODELO MULTI-CLASSE OTIMIZADO (COMPLEMENT NAIVE BAYES)
    # =========================================================================
    print("==========================================================")
    print(" 2. MODELO MULTI-CLASSE (COMPLEMENT NAIVE BAYES - ALPHA 0.05)")
    print("==========================================================")

    # ComplementNB com alpha pequeno para máxima sensibilidade
    modelo_multiclasse = ComplementNB(alpha=0.05)
    modelo_multiclasse.fit(X_treino_tfidf, y_cat_treino)

    y_pred_cat = modelo_multiclasse.predict(X_teste_tfidf)

    nomes_rotulos = ['Neutro [0,0]', 'Ataque Ind. [0,1]', 'Discurso Ódio [1,0]', 'Misto [1,1]']

    print("\n--- Relatório de Desempenho Otimizado (Multi-Classe) ---")
    print(classification_report(y_cat_teste, y_pred_cat, target_names=nomes_rotulos, digits=4))

    print("--- Matriz de Confusão ---")
    matriz_cm = confusion_matrix(y_cat_teste, y_pred_cat)
    matriz_df = pd.DataFrame(matriz_cm, index=nomes_rotulos, columns=nomes_rotulos)
    print(matriz_df)
    print("\n")

    # =========================================================================
    # MODELO MULTI-RÓTULO OTIMIZADO (BINARY RELEVANCE COM MULTINOMIAL/COMPLEMENT)
    # =========================================================================
    print("==========================================================")
    print(" 3. MODELO MULTI-RÓTULO (BINARY RELEVANCE OTIMIZADO)")
    print("==========================================================")

    modelo_multirotulo = MultiOutputClassifier(MultinomialNB(alpha=0.05))
    modelo_multirotulo.fit(X_treino_tfidf, Y_multi_treino)

    Y_pred_multi = modelo_multirotulo.predict(X_teste_tfidf)

    perda_hamming = hamming_loss(Y_multi_teste, Y_pred_multi)
    acuracia_exata = accuracy_score(Y_multi_teste, Y_pred_multi)
    f1_macro = f1_score(Y_multi_teste, Y_pred_multi, average='macro')

    print(f"Hamming Loss (quanto menor, melhor): {perda_hamming:.4f}")
    print(f"Acurácia Exata do Par (Subset Accuracy): {acuracia_exata * 100:.2f}%")
    print(f"F1-Score Macro: {f1_macro:.4f}\n")

    # Salvamento dos Artefatos Otimizados
    pasta_destino = "modelos_salvos"
    os.makedirs(pasta_destino, exist_ok=True)
    
    joblib.dump(vetorizador, os.path.join(pasta_destino, "vetorizador_hibrido.pkl"))
    joblib.dump(modelo_multiclasse, os.path.join(pasta_destino, "naive_bayes_otimizado.pkl"))
    
    print(f"-> Artefatos otimizados salvos na pasta '{pasta_destino}/'!")

    return vetorizador, modelo_multiclasse


def testar_frase_individual(texto, vetorizador, modelo_multiclasse):
    """
    Realiza a inferência e exibe as probabilidades calculadas.
    """
    mapa_categorias = {
        0: "Neutro [0,0]",
        1: "Ataque Individual [0,1]",
        2: "Discurso de Ódio [1,0]",
        3: "Misto [1,1]"
    }
    
    vetor_texto = vetorizador.transform([texto])
    classe_prevista = modelo_multiclasse.predict(vetor_texto)[0]
    
    # Para o ComplementNB, calculamos a distribuição de probabilidade normalizada via Softmax
    log_probs = modelo_multiclasse.predict_log_proba(vetor_texto)[0]
    e_x = np.exp(log_probs - np.max(log_probs))
    probabilidades = e_x / e_x.sum()

    print(f"\nEntrada: \"{texto}\"")
    print(f"Predição Otimizada: {mapa_categorias[classe_prevista]}")
    print("Probabilidades por Classe:")
    for idx, prob in enumerate(probabilidades):
        print(f"  - {mapa_categorias[idx]}: {prob * 100:.2f}%")


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
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
        print("[ERRO] Nenhum arquivo CSV do dataset foi encontrado.")
    else:
        vetorizador_treinado, modelo_treinado = treinar_e_avaliar_otimizado(caminho_final)

        print("\n" + "=" * 60)
        print(" 4. TESTES DE INFERÊNCIA COM O MODELO OTIMIZADO")
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