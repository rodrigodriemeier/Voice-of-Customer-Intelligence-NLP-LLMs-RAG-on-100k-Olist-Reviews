"""Quanto a revisão humana do gold concorda com a primeira passada (feita com LLM)?

A primeira passada está congelada em data/annotations/gold_400_primeira_passada.csv.
Cada linha revisada na tela "Revisar anotações" é comparada com ela.

O número que importa é o das revisões em modo CEGO (quem revisa não vê os
rótulos da primeira passada). Revisão com os rótulos na tela tende a só
confirmar o que já está lá, e a concordância sai inflada.

Também recalcula as métricas principais só nas avaliações revisadas, usando
as predições já salvas (não precisa retreinar nem chamar o LLM de novo).

Rodar:  python -m src.eval_gold
"""
import json
import warnings

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import ANNOTATIONS, ARTIFACTS, CATEGORY_KEYS, SEVERITIES, score_to_sentiment
from src.gold import load_gold, split_list
from src.metrics_store import load_metrics, save_section
from src.rules import label_categories

FIRST_PASS = ANNOTATIONS / "gold_400_primeira_passada.csv"
MIN_FOR_METRICS = 30  # abaixo disso o F1 balança demais para valer a pena reportar


def _kappa(a, b, **kw):
    # com uma classe só (ex.: nenhum wrong_product na amostra) o kappa não é
    # definido; o sklearn devolve NaN, registrado aqui como None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            k = cohen_kappa_score(a, b, **kw)
        except ValueError:
            return None
    return None if k != k else round(float(k), 3)  # NaN quando só existe uma classe


def _jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def agreement(rev, first):
    m = rev.merge(first, on="review_id", suffixes=("", "_1a"))
    sev_num = {s: i for i, s in enumerate(SEVERITIES)}
    cats_h = m["categories"].map(split_list)
    cats_1 = m["categories_1a"].map(split_list)

    per_cat = {}
    for c in CATEGORY_KEYS:
        h = cats_h.map(lambda x, c=c: c in x)
        f = cats_1.map(lambda x, c=c: c in x)
        per_cat[c] = {"concordancia": round(float((h == f).mean()), 3), "kappa": _kappa(h, f)}

    changed = (
        (m["sentiment"] != m["sentiment_1a"])
        | (m["severity"] != m["severity_1a"])
        | (cats_h.map(sorted) != cats_1.map(sorted))
    )
    return {
        "n": int(len(m)),
        "linhas_com_alguma_mudanca": int(changed.sum()),
        "sentimento": {
            "concordancia": round(float((m["sentiment"] == m["sentiment_1a"]).mean()), 3),
            "kappa": _kappa(m["sentiment"], m["sentiment_1a"]),
        },
        "severidade": {
            "concordancia": round(float((m["severity"] == m["severity_1a"]).mean()), 3),
            # severidade é ordinal: errar "high" por "medium" é menos grave que por "none"
            "kappa_ponderado": _kappa(m["severity"].map(sev_num), m["severity_1a"].map(sev_num), weights="quadratic"),
        },
        "categorias": {
            "conjunto_identico": round(float((cats_h.map(sorted) == cats_1.map(sorted)).mean()), 3),
            "jaccard_medio": round(float(pd.Series([_jaccard(a, b) for a, b in zip(cats_h, cats_1)]).mean()), 3),
            "por_categoria": per_cat,
        },
    }


def _load_llm_preds():
    path = ARTIFACTS / "gold_pred_llm.jsonl"
    if not path.exists():
        return None
    model = load_metrics().get("llm_extracao", {}).get("modelo")
    preds = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        if model and p.get("model") != model:
            continue
        if "error" in p:
            p = {**p, "sentiment": "neutral", "categories": [], "severity": "none"}
        preds[p["review_id"]] = p
    return preds


def subset_metrics(rev):
    """Macro-F1 só nas linhas revisadas, para cada método que tiver predição salva."""
    mlb = MultiLabelBinarizer(classes=CATEGORY_KEYS)
    Y = mlb.fit_transform(rev["categories_list"])

    def scores(sent, cats, sev):
        out = {}
        if sent is not None:
            out["sentimento"] = round(f1_score(rev["sentiment"], sent, average="macro"), 3)
        if cats is not None:
            out["categorias"] = round(f1_score(Y, mlb.transform(cats), average="macro", zero_division=0), 3)
        if sev is not None:
            out["severidade"] = round(f1_score(rev["severity"], sev, average="macro", labels=SEVERITIES), 3)
        return out

    res = {
        "Só a nota": scores(rev["review_score"].map(score_to_sentiment), None, None),
        "Regras": scores(None, [label_categories(t) for t in rev["review_text"]], None),
    }
    tf_path = ARTIFACTS / "gold_pred_tfidf.csv"
    if tf_path.exists():
        tf = pd.read_csv(tf_path, keep_default_na=False).set_index("review_id").loc[rev["review_id"]]
        res["TF-IDF + LogReg"] = scores(tf["sentiment"].tolist(), tf["categories"].map(split_list).tolist(),
                                        tf["severity"].tolist())
    llm = _load_llm_preds()
    if llm and all(r in llm for r in rev["review_id"]):
        ps = [llm[r] for r in rev["review_id"]]
        res["LLM"] = scores([p["sentiment"] for p in ps], [p["categories"] for p in ps], [p["severity"] for p in ps])
    return res


def refresh_classifier_metrics(gold):
    """Recalcula as métricas do TF-IDF no gold com os rótulos atuais.

    O treino não usa o gold (ele fica de fora), então revisar rótulos não exige
    retreinar: basta reavaliar o modelo salvo. Sem isso a tabela principal
    continuaria mostrando os números da primeira passada.
    """
    from src.classifier import MODEL_FILE, ReviewClassifier, evaluate_on_gold

    section = load_metrics().get("classificador")
    if not section or not MODEL_FILE.exists():
        return
    section["gold"], pred = evaluate_on_gold(ReviewClassifier.load(), gold)
    pred.insert(0, "review_id", gold["review_id"])
    pred["categories"] = pred["categories"].map("|".join)
    pred.to_csv(ARTIFACTS / "gold_pred_tfidf.csv", index=False)
    save_section("classificador", {k: v for k, v in section.items() if not k.startswith("_")})


def main():
    gold = load_gold()
    first = pd.read_csv(FIRST_PASS, keep_default_na=False, dtype=str)
    rev = gold[gold["reviewed"]].reset_index(drop=True)
    blind = rev[rev["review_mode"] == "cega"].reset_index(drop=True)

    out = {"n_total": int(len(gold)), "n_revisadas": int(len(rev)), "n_revisadas_cegas": int(len(blind))}
    if len(blind):
        out["concordancia_cega"] = agreement(blind, first)
    if len(rev) - len(blind):
        out["concordancia_com_rotulos_visiveis"] = agreement(rev[~rev["review_id"].isin(blind["review_id"])], first)
    if len(rev) >= MIN_FOR_METRICS:
        out["metricas_so_revisadas"] = subset_metrics(rev)

    save_section("revisao_gold", out)
    refresh_classifier_metrics(gold)
    print(f"{len(rev)} de {len(gold)} revisadas ({len(blind)} em modo cego)")
    if "concordancia_cega" in out:
        a = out["concordancia_cega"]
        print(f"  cega: sentimento κ={a['sentimento']['kappa']} | severidade κw={a['severidade']['kappa_ponderado']} "
              f"| categorias idênticas {a['categorias']['conjunto_identico']:.0%}")
    if "metricas_so_revisadas" in out:
        print(pd.DataFrame(out["metricas_so_revisadas"]).T.to_string())


if __name__ == "__main__":
    main()
