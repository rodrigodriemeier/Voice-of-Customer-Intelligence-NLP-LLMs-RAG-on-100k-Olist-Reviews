"""Embeddings das avaliações.

Dois backends:
  - sbert: Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2 por
           padrão). É o que deve ser usado. Baixa ~470 MB na primeira vez.
  - lsa:   TF-IDF + SVD truncado. Não é "semântico" de verdade, mas pega
           sinônimos que co-ocorrem e roda sem internet e sem torch.
           Existe como plano B e como baseline na avaliação de busca.

Se o sbert não puder ser carregado (sem internet ou sem sentence-transformers
instalado), o lsa é usado no lugar, com aviso.
"""
import json
import warnings

import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from src.config import EMBEDDING_BACKEND, RANDOM_STATE, SBERT_MODEL
from src.text import normalize


class SbertEmbedder:
    backend = "sbert"

    def __init__(self, model_name=SBERT_MODEL):
        from sentence_transformers import SentenceTransformer  # import pesado, só se precisar
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # o nome do método que devolve a dimensão mudou entre versões da lib
        # (get_sentence_embedding_dimension -> get_embedding_dimension);
        # medir direto evita depender de qualquer um dos dois
        self.dim = int(self.encode(["teste"]).shape[1])

    def encode(self, texts, batch_size=64, show_progress=False):
        emb = self.model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=show_progress, convert_to_numpy=True,
        )
        return emb.astype("float32")

    def describe(self):
        return {"backend": self.backend, "model": self.model_name, "dim": self.dim}

    def save(self, folder):
        (folder / "embedder.json").write_text(json.dumps(self.describe()), encoding="utf-8")


class LsaEmbedder:
    backend = "lsa"

    def __init__(self, dim=200):
        self.dim = dim
        self.pipe = None

    def fit(self, texts):
        # vocabulário limitado: com bigramas sem limite o arquivo salvo passa de 70 MB
        self.pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, max_features=20_000, sublinear_tf=True),
            TruncatedSVD(n_components=self.dim, random_state=RANDOM_STATE),
            Normalizer(copy=False),
        )
        self.pipe.fit([normalize(t) for t in texts])
        svd = self.pipe.named_steps["truncatedsvd"]
        svd.components_ = svd.components_.astype("float32")
        return self

    def encode(self, texts, batch_size=None, show_progress=False):
        return self.pipe.transform([normalize(t) for t in texts]).astype("float32")

    def describe(self):
        return {"backend": self.backend, "model": f"tfidf+svd{self.dim}", "dim": self.dim}

    def save(self, folder):
        (folder / "embedder.json").write_text(json.dumps(self.describe()), encoding="utf-8")
        joblib.dump(self.pipe, folder / "lsa.joblib", compress=3)

    @classmethod
    def load(cls, folder):
        info = json.loads((folder / "embedder.json").read_text(encoding="utf-8"))
        obj = cls(dim=info["dim"])
        obj.pipe = joblib.load(folder / "lsa.joblib")
        return obj


def build_embedder(fit_texts, backend=None):
    """Cria o embedder para indexar. `fit_texts` só é usado pelo lsa."""
    backend = (backend or EMBEDDING_BACKEND).lower()
    if backend == "sbert":
        try:
            return SbertEmbedder()
        except Exception as err:
            warnings.warn(f"Não consegui carregar o Sentence Transformers ({err}). Usando LSA.")
    return LsaEmbedder().fit(fit_texts)


def load_embedder(folder):
    """Recria o embedder que foi usado para montar um índice salvo."""
    info = json.loads((folder / "embedder.json").read_text(encoding="utf-8"))
    if info["backend"] == "sbert":
        return SbertEmbedder(info["model"])
    return LsaEmbedder.load(folder)

