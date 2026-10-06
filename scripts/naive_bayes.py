#imports;
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    hamming_loss,
    multilabel_confusion_matrix
)

def carregar_e_preparar_dados(caminho_csv):
    """
    Carrega o dataset consolidado e estrutura as variáveis de entrada (X) 
    e alvos (y) tanto para multi-classe (0 a 3) quanto para multi-rótulo ([G, I]).
    """
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo não encontrado em: {caminho_csv}")

    df = pd.read_csv(caminho_csv)
    
    # Validação das colunas esperadas
    colunas_obrigatorias = {'texto', 'ataque_grupo', 'ataque_individual'}
    if not colunas_obrigatorias.issubset(df.columns):
        raise ValueError(f"O CSV deve conter as colunas: {colunas_obrigatorias}")

    # Remove linhas nulas no texto
    df = df.dropna(subset=['texto']).reset_index(drop=True)

    # Mapeamento Multi-Classe de 4 Categorias:
    # [0, 0] -> 0 (Neutro)
    # [0, 1] -> 1 (Ataque Individual)
    # [1, 0] -> 2 (Discurso de Ódio)
    # [1, 1] -> 3 (Misto)
    def mapear_categoria(row):
        g, i = row['ataque_grupo'], row['ataque_individual']
        if g == 0 and i == 0:
            return 0
        elif g == 0 and i == 1:
            return 1
        elif g == 1 and i == 0:
            return 2
        elif g == 1 and i == 1:
            return 3
        return 0

    df['classe_4cat'] = df.apply(mapear_categoria, axis=1)

    X = df['texto']
    y_multirotulo = df[['ataque_grupo', 'ataque_individual']].values
    y_multiclasse = df['classe_4cat'].values

    return X, y_multirotulo, y_multiclasse


def treinar_e_avaliar_naive_bayes(caminho_dataset):
    # 1. Carregamento dos Dados
    X, Y_multi, y_cat = carregar_e_preparar_dados(caminho_dataset)

    # 2. Divisão Treino/Teste (80/20) Estratificada pela classe de 4 categorias
    X_train, X_test, y_cat_train, y_cat_test, Y_multi_train, Y_multi_test = train_test_split(
        X, y_cat, Y_multi,
        test_size=0.20,
        random_state=42,
        stratify=y_cat
    )

    print(f"=== ESTATÍSTICAS DA DIVISÃO ===")
    print(f"Total de amostras de Treino: {len(X_train)}")
    print(f"Total de amostras de Teste:  {len(X_test)}\n")

    # 3. Vetorização TF-IDF (Unigramas e Bigramas com suporte a stop-words em Português)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.90
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # =========================================================================
    # ABORDAGEM A: MODELO MULTI-CLASSE (4 CLASSES DIRETAS)
    # =========================================================================
    print("=" * 60)
    print("  ABORDAGEM 1: NAIVE BAYES MULTI-CLASSE (4 CATEGORIAS)")
    print("=" * 60)

    # MultinomialNB com Suavização de Laplace (alpha=0.1 ou 1.0)
    model_multiclass = MultinomialNB(alpha=0.5)
    model_multiclass.fit(X_train_tfidf, y_cat_train)

    y_cat_pred = model_multiclass.predict(X_test_tfidf)

    nomes_classes = ['Neutro [0,0]', 'Ataque Ind. [0,1]', 'Discurso Ódio [1,0]', 'Misto [1,1]']
    
    print("\n--- Relatório de Classificação (4 Classes) ---")
    print(classification_report(y_cat_test, y_cat_pred, target_names=nomes_classes, digits=4))

    print("--- Matriz de Confusão ---")
    cm = confusion_matrix(y_cat_test, y_cat_pred)
    cm_df = pd.DataFrame(cm, index=nomes_classes, columns=nomes_classes)
    print(cm_df)
    print("\n")

    # =========================================================================
    # ABORDAGEM B: MODELO MULTI-RÓTULO (BINARY RELEVANCE)
    # =========================================================================
    print("=" * 60)
    print("  ABORDAGEM 2: NAIVE BAYES MULTI-RÓTULO (BINARY RELEVANCE)")
    print("=" * 60)

    model_multilabel = MultiOutputClassifier(MultinomialNB(alpha=0.5))
    model_multilabel.fit(X_train_tfidf, Y_multi_train)

    Y_multi_pred = model_multilabel.predict(X_test_tfidf)

    # Métricas de Multi-Rótulo
    h_loss = hamming_loss(Y_multi_test, Y_multi_pred)
    subset_acc = accuracy_score(Y_multi_test, Y_multi_pred)
    f1_macro = f1_score(Y_multi_test, Y_multi_pred, average='macro')
    f1_micro = f1_score(Y_multi_test, Y_multi_pred, average='micro')

    print(f"\nHamming Loss (menor é melhor): {h_loss:.4f}")
    print(f"Exact Match Ratio / Subset Accuracy: {subset_acc * 100:.2f}%")
    print(f"F1-Score Macro: {f1_macro:.4f}")
    print(f"F1-Score Micro: {f1_micro:.4f}\n")

    print("--- Detalhamento por Rótulo Binário ---")
    print("1. Rótulo 'ataque_grupo':")
    print(classification_report(Y_multi_test[:, 0], Y_multi_pred[:, 0], digits=4))

    print("2. Rótulo 'ataque_individual':")
    print(classification_report(Y_multi_test[:, 1], Y_multi_pred[:, 1], digits=4))

    # =========================================================================
    # SALVANDO ARTEFATOS DO MODELO
    # =========================================================================
    os.makedirs("modelos_salvos", exist_ok=True)
    joblib.dump(vectorizer, "modelos_salvos/tfidf_vectorizer.pkl")
    joblib.dump(model_multiclass, "modelos_salvos/naive_bayes_multiclasse.pkl")
    joblib.dump(model_multilabel, "modelos_salvos/naive_bayes_multirotulo.pkl")
    print("Modelos e Vetorizador salvos em 'modelos_salvos/' com sucesso!")

    return vectorizer, model_multiclass


def predizer_novo_texto(texto, vectorizer, model_multiclass):
    """
    Função utilitária para testar o modelo treinado em frases inéditas.
    """
    m_classes = {0: "Neutro [0,0]", 1: "Ataque Individual [0,1]", 2: "Discurso de Ódio [1,0]", 3: "Misto [1,1]"}
    vetor = vectorizer.transform([texto])
    pred = model_multiclass.predict(vetor)[0]
    probas = model_multiclass.predict_proba(vetor)[0]
    
    print(f"\nTexto: '{texto}'")
    print(f"Predição: {m_classes[pred]}")
    print("Probabilidades por Classe:")
    for idx, prob in enumerate(probas):
        print(f"  - {m_classes[idx]}: {prob * 100:.2f}%")


if __name__ == "__main__":
    # Ajuste o caminho de entrada de acordo com seu repositório
    CAMINHO_DATASET = "Dataset_criado/dataset_consolidado.csv"
    
    # Se o arquivo consolidado estiver na pasta raiz para testes locais:
    if not os.path.exists(CAMINHO_DATASET):
        CAMINHO_DATASET = "dataset_consolidado.csv"

    # Executa treino e avaliação
    vec, model = treinar_e_avaliar_naive_bayes(CAMINHO_DATASET)

    # Exemplo prático de inferência para validação
    print("\n" + "="*60)
    print("  TESTE DE INFERÊNCIA EM TEMPO REAL")
    print("="*60)
    
    predizer_novo_texto("Acho que deveríamos estudar mais sobre inteligência artificial.", vec, model)
    predizer_novo_texto("Você é um imbecil e não sabe do que está falando.", vec, model)
    predizer_novo_texto("Esses policiais são todos uns porcos corruptos.", vec, model)
    predizer_novo_texto("Seu viado de merda, sai daqui.", vec, model)