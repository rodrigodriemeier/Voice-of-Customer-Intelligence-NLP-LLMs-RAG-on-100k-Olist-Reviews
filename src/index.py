"""Etapa 3a: gera os embeddings das avaliações e monta o índice FAISS.

Rodar:  python -m src.index            (usa EMBEDDING_BACKEND, padrão sbert)
        python -m src.index --backend lsa

Uso IndexScalarQuantizer de 8 bits em vez do IndexFlat: o índice fica 4x
menor (~16 MB com o MiniLM em vez de ~65 MB), o que importa para subir no
GitHub / Streamlit Cloud. Em comparação com o IndexFlatIP nas 15 consultas
do eval_search, 98% do top-10 coincide em média (pior caso: 8 de 10).
"""
import argparse
import time

import faiss
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, DATA_PROCESSED
from src.embeddings import build_embedder, load_embedder

INDEX_DIR = ARTIFACTS / "index"

META_COLS = [
    "review_id", "order_id", "review_score", "review_text", "review_creation_date",
    "product_category", "macro_category", "customer_state", "delay_days", "is_late",
    "was_delivered", "pred_sentiment", "pred_categories", "pred_severity",
]


def build_index(backend=None):
    corpus = pd.read_parquet(DATA_PROCESSED / "corpus_scored.parquet")
    texts = corpus["review_text"].tolist()

    t0 = time.time()
    embedder = build_embedder(texts, backend=backend)
    print(f"Embedder: {embedder.describe()}")
    vecs = embedder.encode(texts, batch_size=64, show_progress=True)
    print(f"   {len(vecs):,} vetores de dimensão {vecs.shape[1]} em {time.time() - t0:.0f}s")

    index = faiss.IndexScalarQuantizer(vecs.shape[1], faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "reviews.faiss"))
    corpus[META_COLS].to_parquet(INDEX_DIR / "meta.parquet", index=False)
    embedder.save(INDEX_DIR)
    print(f"Índice salvo em {INDEX_DIR}")


class ReviewIndex:
    """Índice + metadados + embedder, carregados juntos."""

    def __init__(self, folder=INDEX_DIR):
        self.folder = folder
        self.index = faiss.read_index(str(folder / "reviews.faiss"))
        self.meta = pd.read_parquet(folder / "meta.parquet")
        self.embedder = load_embedder(folder)

    @property
    def backend(self):
        return self.embedder.describe()

    def search(self, query, k=8, mask=None):
        """Busca semântica. `mask` é um array booleano do tamanho do corpus
        (vem dos filtros da interface); só as linhas True podem voltar."""
        qv = self.embedder.encode([query])
        params = None
        if mask is not None:
            ids = np.flatnonzero(np.asarray(mask))
            if len(ids) == 0:
                return self.meta.head(0).assign(similarity=[])
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids.astype("int64")))
            k = min(k, len(ids))
        scores, idx = self.index.search(qv, k, params=params)
        keep = idx[0] >= 0
        hits = self.meta.iloc[idx[0][keep]].copy()
        # a quantização de 8 bits pode devolver 1.0002; limita a [-1, 1]
        hits["similarity"] = np.clip(scores[0][keep], -1.0, 1.0)
        return hits.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sbert", "lsa"], default=None)
    build_index(ap.parse_args().backend)
