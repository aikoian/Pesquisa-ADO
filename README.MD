# [Nome do Projeto] - Análise de Discurso de Ódio com LLMs

[![Status do Projeto](https://img.shields.io/badge/Status-Em_Andamento-blue.svg)](https://shields.io/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/Dataset_License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

Este repositório documenta um projeto de extensão focado na análise e detecção de discurso de ódio nas redes sociais (com foco inicial no Reddit) para o português brasileiro, utilizando Modelos de Linguagem de Grande Porte (LLMs).

O principal artefato deste projeto é um **novo dataset de 2.428 amostras**, criado para permitir uma classificação mais granular de toxicidade, diferenciando ataques individuais de discurso de ódio direcionado a grupos.

## 1. O Dataset: Motivação e Taxonomia

Datasets de referência para o português, como o **ToLD-BR** e o **OLID-BR**, são fundamentais, mas frequentemente agrupam diferentes tipos de toxicidade (ofensa e ódio) em um único rótulo. Isso pode ser uma limitação ao treinar modelos que precisam de uma compreensão mais precisa e contextual.

Para resolver essa lacuna, propomos uma taxonomia multi-rótulo de 4 classes que desambigua **Ataques Individuais** de **Discurso de Ódio (Ataques a Grupos)**.

Nosso dataset é perfeitamente balanceado, contendo **607 amostras** para cada uma das seguintes classes:

| `ataque_grupo` | `ataque_individual` | Categoria | Descrição |
| :--- | :--- | :--- | :--- |
| `0` | `0` | **Neutro** | Texto não-ofensivo. |
| `0` | `1` | **Ataque Individual** | Ofensa ou xingamento direcionado a um indivíduo. |
| `1` | `0` | **Discurso de Ódio** | Ataque generalizado a um grupo protegido. |
| `1` | `1` | **Misto** | Ataque individual que utiliza um termo de ódio a grupo. |

## 2. Metodologia de Construção

Para garantir um dataset de alta qualidade e autenticidade, adotamos uma metodologia híbrida:

1.  **Geração Assistida por LLM:** Uso de prompts detalhados para gerar frases-base, seguido de uma **pós-edição manual intensiva** para remover vícios de IA e aumentar a verossimilhança.
2.  **Síntese Manual (Classe Mista):** Criação manual da classe `[1,1]` através da fusão e edição coesa de amostras das classes `[1,0]` e `[0,1]`.
3.  **Curadoria de Datasets Existentes:** Revisão linha por linha e reaproveitamento de amostras (principalmente neutras) dos datasets ToLD-BR e OLID-BR.
4.  **Coleta "In-the-Wild":** Coleta suplementar de dados reais via API do Reddit (PRAWN) com posterior análise e classificação manual.

## 3. Status do Projeto

-   [x] Definição da Taxonomia de 4 classes
-   [x] Coleta, Geração e Curadoria do Dataset (2.428 amostras)
-   [x] Pré-processamento (Limpeza, Remoção de Stopwords, Lematização)
-   [ ] Treinamento e Avaliação de LLMs (Em Andamento)
-   [ ] Redação do Artigo Científico (Em Andamento)

## 4. Estrutura do Repositório (Provisório)