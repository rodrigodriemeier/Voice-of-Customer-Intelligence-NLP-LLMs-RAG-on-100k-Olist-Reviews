# Resultados

Arquivo gerado por `python -m src.report` a partir de `reports/metrics.json`. Não edite à mão.

## Classificação

| Macro-F1 no gold (n=400) | Só a nota | Regras | TF-IDF + LogReg | LLM |
|:--|--:|--:|--:|--:|
| Sentimento | 0,70 | — | 0,75 | 0,84 |
| Categoria da reclamação | — | 0,70 | 0,69 | 0,90 |
| Severidade | — | — | 0,56 | 0,79 |

### Por categoria

| Categoria | n | Regras F1 | TF-IDF P | TF-IDF R | TF-IDF F1 | LLM P | LLM R | LLM F1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Entrega | 91 | 0,79 | 0,79 | 0,77 | 0,78 | 0,91 | 0,93 | 0,92 |
| Qualidade / produto danificado | 65 | 0,74 | 0,93 | 0,63 | 0,75 | 0,97 | 0,89 | 0,93 |
| Produto errado ou diferente do anúncio | 37 | 0,55 | 0,80 | 0,43 | 0,56 | 0,80 | 1,00 | 0,89 |
| Pedido incompleto / item faltando | 34 | 0,77 | 0,69 | 0,79 | 0,74 | 0,92 | 0,97 | 0,94 |
| Atendimento e pós-venda | 50 | 0,64 | 0,61 | 0,68 | 0,64 | 0,75 | 0,88 | 0,81 |

Holdout aleatório (6.296 avaliações, rótulo vindo da nota): macro-F1 de sentimento 0,67; F1 por classe: negativo 0,81, neutro 0,32, positivo 0,90.

## Conjunto de referência

**Revisão humana do gold:** 100 de 400 revisadas, 100 em modo cego (sem ver os rótulos da primeira passada).

| Concordância revisão cega x primeira passada | Resultado |
|:--|--:|
| Sentimento (kappa) | 0,82 |
| Severidade (kappa ponderado) | 0,84 |
| Categorias: conjunto idêntico | 88% |
| Linhas com alguma mudança | 39 de 100 |

Macro-F1 só nas 100 revisadas:

| Método | Sentimento | Categorias | Severidade |
|:--|--:|--:|--:|
| Só a nota | 0,69 | — | — |
| Regras | — | 0,76 | — |
| TF-IDF + LogReg | 0,77 | 0,76 | 0,55 |
| LLM | 0,84 | 0,87 | 0,60 |

## Extração com LLM, campo a campo

Modelo: `claude-haiku-5-5`

| Campo | Resultado |
|:--|--:|
| Sentimento (macro-F1) | 0,84 |
| Categorias (micro P / R) | 0,87 / 0,93 |
| Severidade (macro-F1) | 0,79 |
| Aspectos (micro P / R) | 0,74 / 0,95 |
| Evidência copiada literalmente do texto | 100% |
| Evidência x anotação (sobreposição de tokens) | 0,85 |
| Detectou que há problema (P / R) | 0,97 / 0,99 |
| Respostas sem JSON válido | 0 de 400 |

## Busca semântica

| Método | Recall@20 | Recall@50 | Recall@100 | P@10 |
|:--|--:|--:|--:|--:|
| TF-IDF (palavra-chave) | 0,14 | 0,27 | 0,41 | 0,48 |
| LSA (TF-IDF + SVD) | 0,20 | 0,34 | 0,55 | 0,55 |
| Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2) | 0,19 | 0,38 | 0,56 | 0,59 |
| Aleatório (referência) | 0,05 | 0,12 | 0,25 | 0,14 |

**Índice de busca incluído no repositório:** Sentence Transformers (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).

## RAG

12 perguntas · gerador `claude-haiku-5-5` · juiz `claude-haiku-5-5` · embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

| Métrica | Resultado |
|:--|--:|
| Faithfulness (afirmações sustentadas pelas avaliações citadas) | 0,85 |
| Relevância do contexto (avaliações recuperadas que têm a ver) | 0,79 |
| Relevância da resposta (1 a 5) | 3,50 |
| Trechos da resposta com citação | 0,42 |
| Respostas com número que não veio do Python | 0 de 12 |
