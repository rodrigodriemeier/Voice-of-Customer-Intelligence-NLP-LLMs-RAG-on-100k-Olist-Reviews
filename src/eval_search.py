"""Avaliação da busca semântica (etapa 3).

Corpus de teste: as 400 avaliações do gold (a única base com rótulo).
Consultas: 3 frases por categoria de reclamação, de propósito escritas de
jeitos diferentes: uma direta, uma mais formal e uma que não usa a palavra
óbvia ("a encomenda sumiu no caminho" em vez de "atraso").
Relevante = avaliação anotada com aquela categoria.

Métricas: Recall@20/50/100 e P@10. Como cada categoria tem de 34 a 96
avaliações relevantes, Recall@10 seria limitado por construção (com 96
relevantes, o máximo seria 0,10), por isso os K maiores. Como referência,
a tabela inclui o resultado esperado de uma ordenação aleatória.

Um teste "known-item" (usar a descrição do problema como consulta e
recuperar a própria avaliação entre as 42 mil) não é informativo aqui:
existem milhares de "não recebi o produto" praticamente iguais, e qualquer
um deles é uma resposta correta.

Rodar:  python -m src.eval_search
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2

from src.config import CATEGORY_KEYS, DATA_PROCESSED
from src.embeddings import LsaEmbedder, SbertEmbedder
from src.gold import load_gold
from src.metrics_store import save_section
from src.text import normalize

KS = (20, 50, 100)

QUERIES = {
    "delivery": [
        "meu pedido está atrasado e até agora não chegou",
        "problemas com o prazo de entrega",
        "a encomenda sumiu no caminho",
    ],
    "product_quality": [
        "o produto veio quebrado e com defeito",
        "qualidade ruim do material",
        "parou de funcionar logo depois que chegou",
    ],
    "wrong_product": [
        "recebi um produto diferente do que eu comprei",
        "não é igual à foto do anúncio",
        "mandaram a cor e o modelo errados",
    ],
    "missing_items": [
        "comprei duas unidades e só veio uma",
        "o pedido chegou incompleto",
        "faltou uma peça dentro da caixa",
    ],
    "customer_service": [
        "a loja não responde e não consigo o reembolso",
        "dificuldade para cancelar ou devolver a compra",
        "atendimento péssimo, ninguém resolve nada",
    ],
}


class TfidfSearch:
    """Busca por palavra-chave, o baseline."""

    def fit(self, texts):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.vec.fit([normalize(t) for t in texts])
        return self

    def encode(self, texts, **_):
        return l2(self.vec.transform([normalize(t) for t in texts])).toarray().astype("float32")


def get_methods(fit_texts):
    """O TF-IDF e o LSA são ajustados no corpus inteiro (como seriam em produção),
    não só nas 400 do gold, senão o vocabulário fica pobre demais."""
    methods = {
        "TF-IDF (palavra-chave)": TfidfSearch().fit(fit_texts),
        "LSA (TF-IDF + SVD)": LsaEmbedder().fit(fit_texts),
    }
    try:
        sb = SbertEmbedder()
        methods[f"Sentence Transformers ({sb.model_name.split('/')[-1]})"] = sb
    except Exception as err:
        print(f"   Sentence Transformers indisponível: {type(err).__name__}: {str(err)[:150]}")
    return methods


def evaluate_method(method, gold):
    docs = method.encode(gold["review_text"].tolist())
    per_cat = {}
    for cat in CATEGORY_KEYS:
        rel = gold["categories_list"].map(lambda cs: cat in cs).to_numpy()
        n_rel = int(rel.sum())
        rows = []
        for query in QUERIES[cat]:
            q = method.encode([query])[0]
            order = np.argsort(-(docs @ q))
            row = {f"recall@{k}": rel[order[:k]].sum() / n_rel for k in KS}
            row["p@10"] = rel[order[:10]].mean()
            rows.append(row)
        per_cat[cat] = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
        per_cat[cat]["n_relevantes"] = n_rel
    media = {k: round(float(np.mean([v[k] for v in per_cat.values()])), 3)
             for k in [f"recall@{k}" for k in KS] + ["p@10"]}
    return {"media": media, "por_categoria": per_cat}


def main():
    corpus = pd.read_parquet(DATA_PROCESSED / "corpus.parquet", columns=["review_text"])
    gold = load_gold()
    methods = get_methods(corpus["review_text"].tolist())

    results = {name: evaluate_method(m, gold) for name, m in methods.items()}
    # referência: ordem aleatória. Recall@K esperado = K/N; P@10 = prevalência
    prev = [gold["categories_list"].map(lambda cs, c=c: c in cs).mean() for c in CATEGORY_KEYS]
    results["Aleatório (referência)"] = {"media": {
        **{f"recall@{k}": round(k / len(gold), 3) for k in KS},
        "p@10": round(float(np.mean(prev)), 3),
    }}

    print("\nBusca por categoria no gold (média de 15 consultas):")
    print(pd.DataFrame({k: v["media"] for k, v in results.items()}).T.to_string())

    save_section("busca", {
        "n_docs": int(len(gold)),
        "n_consultas": sum(len(v) for v in QUERIES.values()),
        "consultas": QUERIES,
        "resultados": results,
    })


if __name__ == "__main__":
    main()
