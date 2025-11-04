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
```/│ ├── Dataset_criado/ │ ├── dataset_gerado/ │ │ ├── dataset_formatado_bruto.csv (O dataset original) │ │ └── dataset_formatado_tratado.csv (O dataset pré-processado) │ │ │ ├── olid_gerados_[0,0]/ (Amostras-fonte usadas para a classe) │ ├── olid_gerados_[0,1]/ (Amostras-fonte usadas para a classe) │ ├── olid_gerados_[1,0]/ (Amostras-fonte usadas para a classe) │ └── olid_gerados_[1,1]/ (Amostras-fonte usadas para a classe) │ ├── scripts/ (Scripts de coleta, processamento, etc.) └── README.md (Este arquivo)```


## 5. Licença e Atribuição

### Licença do Dataset

O dataset (`/Dataset_criado/dataset_gerado/`) contido neste repositório é um **trabalho derivado** que incorpora materiais dos datasets ToLD-BR e OLID-BR.

Devido à licença "CompartilhaIgual" (SA) do ToLD-BR, este dataset é distribuído sob a mesma licença:
**[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)**.

### Licença do Código-Fonte

O código-fonte (scripts em Python) neste repositório é distribuído sob a **Licença MIT** (a menos que indicado de outra forma).

### Citação (Obrigatório)

Se você utilizar este dataset em seu trabalho, por favor, cite nosso artigo (ainda a ser publicado) e os trabalhos originais que formaram sua base.

**Nosso Trabalho (Placeholder):**
```bibtex
@misc{seu_projeto_2025,
  author    = {[Aiko Ian], [Guilherme Leite], [Carol Medeiros], [Augusto Ferrer]},
  title     = {[Título do seu Artigo/Dataset, ex: Um Dataset Granular para Detecção de Ódio e Ofensa em Português]},
  year      = {2025},
  publisher = {GitHub},
  journal   = {GitHub repository},
  howpublished = {\url{[https://github.com/guilhermelbr/Pesquisa-ADO](https://github.com/guilhermelbr/Pesquisa-ADO)}}


## Datasets Originais (Base deste trabalho):

@inproceedings{leite-etal-2020-toxic,
    title = "{T}oxic {L}anguage {D}etection in {S}ocial {M}edia for {B}razilian {P}ortuguese: {N}ew {D}ataset and {M}ultilingual {A}nalysis",
    author = "Leite, Jo{\~a}o Augusto  and
      Silva, Diego  and
      Bontcheva, Kalina  and
      Scarton, Carolina",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "[https://aclanthology.org/2020.aacl-main.91](https://aclanthology.org/2020.aacl-main.91)",
    pages = "914--924",


@article{trajano-etal-2023-olidbr,
    title = "{OLID-BR}: {O}ffensive {L}anguage {I}dentification {D}ataset for {B}razilian {P}ortuguese",
    author = "Trajano, Douglas  and
      Bordini, Rafael H.  and
      Vieira, Renata",
    journal = {Language Resources and Evaluation},
    year = "2023",
    month = may,
    doi = {10.1007/s10579-023-09657-0},
    url = "[https://doi.org/10.1007/s10579-023-09657-0](https://doi.org/10.1007/s10579-023-09657-0)"
}