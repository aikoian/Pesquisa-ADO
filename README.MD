```markdown
# Detecção de Discurso de Ódio em Português: Uma Abordagem Universalista e Granular

![Status do Projeto](https://img.shields.io/badge/Status-Em_Andamento-blue.svg)
![License: CC BY-SA 4.0](https://img.shields.io/badge/Dataset_License-CC_BY--SA_4.0-lightgrey.svg)

Este repositório documenta uma pesquisa de extensão dedicada à análise, detecção e classificação de toxicidade em redes sociais, com foco inicial no português brasileiro. O projeto introduz uma mudança de paradigma ao adotar uma **Abordagem Universalista** e o **Princípio da Prioridade do Alvo**, permitindo uma distinção clara entre conflitos interpessoais (cyberbullying) e ódio coletivo.

## 1. Fundamentação Teórica e Inovações

O diferencial deste projeto reside na superação das limitações conceituais de datasets binários tradicionais através de dois pilares:

### 1.1. Abordagem Universalista (*Universalist Approach*)
Diferente da abordagem restrita que foca exclusivamente em grupos protegidos ou minorias, este projeto assume que a hostilidade coletiva deve ser identificada por sua **estrutura sintática e pragmática**.
* **Definição**: Discurso de Ódio é qualquer manifestação que generalize, desumanize ou incite violência contra **qualquer grupo social** (ex: policiais, políticos, grupos majoritários), não apenas grupos historicamente vulneráveis.
* **Justificativa**: Um modelo robusto deve aprender o padrão de "generalização odiosa" para evitar vieses e garantir uma moderação equânime.

### 1.2. Princípio da Prioridade do Alvo (*Target Priority Principle*)
Para resolver ambiguidades onde termos de ódio são usados em brigas pessoais, aplicamos a prioridade do alvo:
* **Ataque Individual**: Quando o enunciado visa atingir um sujeito específico (ex: *"Você é um [termo de ódio]"*), ele é classificado primariamente como um ataque individual (Classe 3 - Misto).
* **Ataque a Grupo**: Quando o alvo é a coletividade (ex: *"Esses [termo de ódio] não prestam"*), configura-se o discurso de ódio propriamente dito (Classe 2).

## 2. Taxonomia Multi-Rótulo (4 Classes)

Propomos uma matriz taxonômica que desambigua os cenários através de duas variáveis binárias: `ataque_grupo` e `ataque_individual`.

| Classe | Categoria | Config. (G / I) | Descrição Teórica |
| :--- | :--- | :--- | :--- |
| **0** | **Neutro** | `0 / 0` | Texto não ofensivo. Baseline para o modelo. |
| **1** | **Ataque Individual** | `0 / 1` | Ofensa direcionada a um indivíduo sem viés de grupo. |
| **2** | **Discurso de Ódio** | `1 / 0` | Ataque generalizado a um grupo (Abordagem Universalista). |
| **3** | **Misto** | `1 / 1` | Ataque individual que utiliza estigmas de grupo como arma de ofensa. |

## 3. O Dataset

O dataset consolidado possui **2.428 amostras**, rigorosamente balanceadas com 607 exemplos para cada classe.

### Metodologia de Construção:
1.  **Coleta "In-the-Wild"**: Captura de dados reais do **Reddit** via API (PRAW), preservando a linguagem natural, gírias e variações gramaticais.
2.  **Geração e Aumento de Dados**: Uso de LLMs para gerar frases que preencham lacunas teóricas (ex: ataques a maiorias), seguido de edição manual exaustiva.
3.  **Curadoria Manual**: Revisão linha por linha por especialistas para garantir a aderência ao Princípio da Prioridade do Alvo.
4.  **Integração**: Incorporação e reclassificação de amostras selecionadas do ToLD-BR e OLID-BR.

## 4. Estrutura do Repositório

O repositório está organizado para garantir transparência metodológica e facilitar o treinamento de modelos:

```text
/
├── Dataset_criado/
│   ├── Neutro/                # Amostras da classe [0,0]
│   ├── Ataque_Individual/     # Amostras da classe [0,1]
│   ├── Ataque_a_Grupo/        # Amostras da classe [1,0] (Discurso de Ódio)
│   ├── Ataque_Misto/          # Amostras da classe [1,1]
│   └── dataset_consolidado.csv # Versão final pronta para treinamento
├── scripts/
│   ├── coleta_reddit.py       # Script de extração via PRAW
│   ├── tratamento.py          # Pré-processamento e limpeza
│   └── filtro_palavras.py     # Filtragem estratégica de candidatos
└── README.md


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
  author    = {[Aiko Ian], [Guilherme Leite], [Carol Medeiros]},
  title     = {[Título do seu Artigo/Dataset, ex: Um Dataset Granular para Detecção de Ódio e Ofensa em Português]},
  year      = {2025},
  publisher = {GitHub},
  journal   = {GitHub repository},
  howpublished = {\url{[https://github.com/guilhermelbr/Pesquisa-ADO](https://github.com/guilhermelbr/Pesquisa-ADO)}}}
```

## Datasets Originais (Base deste trabalho):
```bibtex
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
```