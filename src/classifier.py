"""Etapa 2A: TF-IDF + regressão logística para sentimento e categoria.

Rótulos de treino:
  - sentimento: vem da nota (1-2 negativo, 3 neutro, 4-5 positivo)
  - categoria: vem das regras de src/rules.py (supervisão fraca)

Os dois rótulos são ruidosos; por isso a avaliação final usa o gold anotado
(rótulo lido do texto, não da nota).

Rodar:  python -m src.classifier
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import ARTIFACTS, CATEGORY_KEYS, DATA_PROCESSED, RANDOM_STATE, SENTIMENTS, score_to_sentiment
from src.gold import load_gold
from src.metrics_store import save_section
from src.rules import label_categories, severity_from_rules
from src.text import normalize

MODEL_FILE = ARTIFACTS / "classifier.joblib"


def make_features():
    # n-gramas de palavra capturam "nao chegou", "veio quebrado"; os de caractere
    # toleram erros de digitação, frequentes no dataset ("entegue", "podruto")
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                 sublinear_tf=True, max_features=150_000)),
    ])


def pick_C(X_tr, y_tr, X_val, y_val, grid=(0.5, 2.0, 8.0)):
    best = None
    for C in grid:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(X_tr, y_tr)
        score = f1_score(y_val, clf.predict(X_val), average="macro")
        print(f"   C={C:<4} macro-F1 val = {score:.3f}")
        if best is None or score > best[1]:
            best = (C, score)
    return best[0]


class ReviewClassifier:
    def __init__(self, features, sentiment_clf, category_clf, mlb):
        self.features = features
        self.sentiment_clf = sentiment_clf
        self.category_clf = category_clf
        self.mlb = mlb

    def predict_frame(self, texts, threshold=0.5):
        norm = [normalize(t) for t in texts]
        X = self.features.transform(norm)
        sent_proba = self.sentiment_clf.predict_proba(X)
        sent = self.sentiment_clf.classes_[sent_proba.argmax(1)]
        cat_proba = self.category_clf.predict_proba(X)
        out = pd.DataFrame({"sentiment": sent, "sentiment_conf": sent_proba.max(1)})
        cats = []
        for row in cat_proba:
            cats.append([c for c, p in zip(self.mlb.classes_, row) if p >= threshold])
        out["categories"] = cats
        for c, col in zip(self.mlb.classes_, cat_proba.T):
            out[f"p_{c}"] = col
        out["severity"] = [
            severity_from_rules(n, c, s, already_normalized=True) for n, c, s in zip(norm, cats, sent)
        ]
        return out

    def predict(self, text):
        """Saída no formato da funcionalidade A do projeto."""
        row = self.predict_frame([text]).iloc[0]
        return {
            "sentiment": row["sentiment"],
            "categories": list(row["categories"]),
            "severity": row["severity"],
            "confidence": {
                "sentiment": round(float(row["sentiment_conf"]), 3),
                **{c: round(float(row[f"p_{c}"]), 3) for c in self.mlb.classes_},
            },
        }

    # As partes são salvas num dict, não a instância: executando com `python -m`,
    # a classe seria serializada como __main__.ReviewClassifier e o arquivo não
    # abriria em outro contexto (app, testes).
    def save(self, path=MODEL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        parts = {"features": self.features, "sentiment_clf": self.sentiment_clf,
                 "category_clf": self.category_clf, "mlb": self.mlb}
        joblib.dump(parts, path, compress=3)

    @staticmethod
    def load(path=MODEL_FILE):
        return ReviewClassifier(**joblib.load(path))


def _multilabel_report(Y_true, Y_pred):
    rep = classification_report(Y_true, Y_pred, target_names=CATEGORY_KEYS, output_dict=True, zero_division=0)
    per_cat = {c: {k: round(rep[c][k], 3) for k in ("precision", "recall", "f1-score", "support")} for c in CATEGORY_KEYS}
    return {
        "macro_f1": round(f1_score(Y_true, Y_pred, average="macro", zero_division=0), 3),
        "micro_f1": round(f1_score(Y_true, Y_pred, average="micro", zero_division=0), 3),
        "por_categoria": per_cat,
    }


def evaluate_on_gold(model, gold):
    pred = model.predict_frame(gold["review_text"].tolist())
    mlb = MultiLabelBinarizer(classes=CATEGORY_KEYS)
    Y = mlb.fit_transform(gold["categories_list"])
    P = mlb.transform(pred["categories"])
    P_rules = mlb.transform([label_categories(t) for t in gold["review_text"]])

    stars = gold["review_score"].map(score_to_sentiment)
    sev_order = ["none", "low", "medium", "high"]

    return {
        "n": int(len(gold)),
        "sentimento": {
            "tfidf_logreg_macro_f1": round(f1_score(gold["sentiment"], pred["sentiment"], average="macro"), 3),
            "tfidf_logreg_acc": round(accuracy_score(gold["sentiment"], pred["sentiment"]), 3),
            "baseline_estrelas_macro_f1": round(f1_score(gold["sentiment"], stars, average="macro"), 3),
            "baseline_estrelas_acc": round(accuracy_score(gold["sentiment"], stars), 3),
            "por_classe": {
                k: round(v["f1-score"], 3)
                for k, v in classification_report(gold["sentiment"], pred["sentiment"], output_dict=True, zero_division=0).items()
                if k in SENTIMENTS
            },
        },
        "categorias": {
            "tfidf_logreg": _multilabel_report(Y, P),
            "baseline_regras": _multilabel_report(Y, P_rules),
        },
        "severidade": {
            "regras_macro_f1": round(f1_score(gold["severity"], pred["severity"], average="macro", labels=sev_order), 3),
            "regras_acc": round(accuracy_score(gold["severity"], pred["severity"]), 3),
        },
    }, pred


def main():
    t0 = time.time()
    corpus = pd.read_parquet(DATA_PROCESSED / "corpus.parquet")
    gold = load_gold()
    # o gold fica fora do treino
    train_df = corpus[~corpus["review_id"].isin(gold["review_id"])].reset_index(drop=True)

    print(f"Rotulando {len(train_df):,} avaliações com as regras...")
    train_df["weak_categories"] = [label_categories(t, already_normalized=True) for t in train_df["text_norm"]]
    coverage = (train_df["weak_categories"].str.len() > 0).mean()
    print(f"   {coverage:.1%} receberam pelo menos uma categoria")

    tr, te = train_test_split(train_df, test_size=0.15, random_state=RANDOM_STATE,
                              stratify=train_df["review_score"])
    tr, val = train_test_split(tr, test_size=0.12, random_state=RANDOM_STATE, stratify=tr["review_score"])

    print("Vetorizando...")
    features = make_features()
    X_tr = features.fit_transform(tr["text_norm"])
    X_val = features.transform(val["text_norm"])
    X_te = features.transform(te["text_norm"])
    print(f"   {X_tr.shape[1]:,} features")

    print("Sentimento (escolhendo C na validação):")
    C_sent = pick_C(X_tr, tr["sentiment_from_score"], X_val, val["sentiment_from_score"])
    sent_clf = LogisticRegression(C=C_sent, class_weight="balanced", max_iter=3000)
    sent_clf.fit(X_tr, tr["sentiment_from_score"])

    print("Categorias (one-vs-rest):")
    mlb = MultiLabelBinarizer(classes=CATEGORY_KEYS)
    Y_tr = mlb.fit_transform(tr["weak_categories"])
    Y_te = mlb.transform(te["weak_categories"])
    cat_clf = OneVsRestClassifier(LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000))
    cat_clf.fit(X_tr, Y_tr)

    model = ReviewClassifier(features, sent_clf, cat_clf, mlb)
    model.save()

    holdout_sent = sent_clf.predict(X_te)
    holdout = {
        "n": int(len(te)),
        "sentimento_vs_nota_macro_f1": round(f1_score(te["sentiment_from_score"], holdout_sent, average="macro"), 3),
        "sentimento_vs_nota_acc": round(accuracy_score(te["sentiment_from_score"], holdout_sent), 3),
        "categorias_vs_regras_macro_f1": round(
            f1_score(Y_te, (cat_clf.predict_proba(X_te) >= 0.5).astype(int), average="macro", zero_division=0), 3
        ),
        "por_classe": {
            k: round(v["f1-score"], 3)
            for k, v in classification_report(te["sentiment_from_score"], holdout_sent, output_dict=True).items()
            if k in SENTIMENTS
        },
    }
    print(f"Holdout ({len(te):,}): macro-F1 sentimento (vs nota) = {holdout['sentimento_vs_nota_macro_f1']}")

    gold_metrics, gold_pred = evaluate_on_gold(model, gold)
    print(f"Gold: macro-F1 sentimento = {gold_metrics['sentimento']['tfidf_logreg_macro_f1']} "
          f"| categorias = {gold_metrics['categorias']['tfidf_logreg']['macro_f1']} "
          f"(regras: {gold_metrics['categorias']['baseline_regras']['macro_f1']})")

    gold_pred.insert(0, "review_id", gold["review_id"])
    gold_pred["categories"] = gold_pred["categories"].map("|".join)
    gold_pred.to_csv(ARTIFACTS / "gold_pred_tfidf.csv", index=False)

    save_section("classificador", {
        "C_sentimento": C_sent,
        "n_treino": int(len(tr)),
        "cobertura_regras": round(float(coverage), 3),
        "holdout": holdout,
        "gold": gold_metrics,
    })

    print("Pontuando o corpus inteiro para o dashboard...")
    scored = model.predict_frame(corpus["review_text"].tolist())
    scored.columns = [f"pred_{c}" if not c.startswith("p_") else c for c in scored.columns]
    scored["pred_categories"] = scored["pred_categories"].map(lambda xs: "|".join(xs))
    out = pd.concat([corpus.reset_index(drop=True), scored], axis=1)
    out.to_parquet(DATA_PROCESSED / "corpus_scored.parquet", index=False)
    print(f"Pronto em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
