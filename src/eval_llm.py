"""Métricas da extração via LLM contra o gold, campo por campo.

- sentiment, severity: macro-F1 (classe única)
- categories, aspects: precision / recall / F1 por rótulo (multirrótulo)
- evidence: (a) % de evidências que existem literalmente no texto
            (b) sobreposição de tokens com a evidência anotada
- problem: precision/recall de "detectou que há um problema"
"""
import re

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import ARTIFACTS, ASPECTS, CATEGORY_KEYS, SEVERITIES
from src.gold import load_gold
from src.metrics_store import save_section
from src.text import strip_accents


def _tokens(s):
    return set(re.findall(r"\w+", strip_accents(s.lower())))


def token_f1(pred, ref):
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    inter = len(p & r)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(p), inter / len(r)
    return 2 * prec * rec / (prec + rec)


def _prf(Y, P, labels):
    p, r, f, s = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
    per = {lab: {"precision": round(p[i], 3), "recall": round(r[i], 3), "f1": round(f[i], 3), "support": int(s[i])}
           for i, lab in enumerate(labels)}
    mp, mr, mf, _ = precision_recall_fscore_support(Y, P, average="micro", zero_division=0)
    return {
        "macro_f1": round(f1_score(Y, P, average="macro", zero_division=0), 3),
        "micro_precision": round(mp, 3),
        "micro_recall": round(mr, 3),
        "micro_f1": round(mf, 3),
        "por_rotulo": per,
    }


def evaluate_llm(preds, model_name="?", save=True):
    gold = load_gold().set_index("review_id")
    rows = []
    n_errors = 0
    n_issues = 0
    for p in preds:
        if "error" in p:
            n_errors += 1
            # conta como "não respondeu nada": tudo vazio
            p = {"review_id": p["review_id"], "sentiment": "neutral", "categories": [], "severity": "none",
                 "aspects": [], "problem": None}
        n_issues += bool(p.get("issues"))
        g = gold.loc[p["review_id"]]
        rows.append((g, p))

    if not rows:
        print("Nenhuma predição para avaliar.")
        return {}

    g_sent = [g["sentiment"] for g, _ in rows]
    p_sent = [p["sentiment"] for _, p in rows]
    g_sev = [g["severity"] for g, _ in rows]
    p_sev = [p["severity"] for _, p in rows]

    mlb_c = MultiLabelBinarizer(classes=CATEGORY_KEYS)
    Yc = mlb_c.fit_transform([g["categories_list"] for g, _ in rows])
    Pc = mlb_c.transform([p["categories"] for _, p in rows])

    mlb_a = MultiLabelBinarizer(classes=ASPECTS)
    Ya = mlb_a.fit_transform([g["aspects_list"] for g, _ in rows])
    Pa = mlb_a.transform([sorted({a["aspect"] for a in p["aspects"]}) for _, p in rows])

    all_ev = [a for _, p in rows for a in p["aspects"] if a.get("evidence")]
    grounded_rate = sum(a["grounded"] for a in all_ev) / len(all_ev) if all_ev else 0.0

    # compara a evidência anotada (da reclamação principal) com a melhor
    # evidência do LLM; só onde há reclamação ("Ótimo!" não tem o que comparar)
    ev_overlap = []
    for g, p in rows:
        if not g["problem"]:
            continue
        evs = [a["evidence"] for a in p["aspects"] if a.get("evidence")]
        ev_overlap.append(max((token_f1(e, g["evidence"]) for e in evs), default=0.0))

    g_has_prob = [bool(g["problem"]) for g, _ in rows]
    p_has_prob = [bool(p.get("problem")) for _, p in rows]
    pp, pr, pf, _ = precision_recall_fscore_support(g_has_prob, p_has_prob, average="binary", zero_division=0)

    metrics = {
        "modelo": model_name,
        "n": len(rows),
        "respostas_sem_json_valido": n_errors,
        "respostas_com_campo_corrigido": n_issues,
        "sentimento_macro_f1": round(f1_score(g_sent, p_sent, average="macro"), 3),
        "severidade_macro_f1": round(f1_score(g_sev, p_sev, average="macro", labels=SEVERITIES), 3),
        "categorias": _prf(Yc, Pc, CATEGORY_KEYS),
        "aspectos": _prf(Ya, Pa, ASPECTS),
        "evidencia": {
            "literal_no_texto": round(grounded_rate, 3),
            "sobreposicao_tokens_media": round(sum(ev_overlap) / len(ev_overlap), 3) if ev_overlap else 0.0,
        },
        "problema_detectado": {"precision": round(pp, 3), "recall": round(pr, 3), "f1": round(pf, 3)},
    }

    print(f"\nLLM ({model_name}) em {len(rows)} avaliações do gold")
    print(f"  sentimento macro-F1:  {metrics['sentimento_macro_f1']}")
    print(f"  categorias macro-F1:  {metrics['categorias']['macro_f1']}")
    print(f"  severidade macro-F1:  {metrics['severidade_macro_f1']}")
    print(f"  aspectos micro P/R:   {metrics['aspectos']['micro_precision']} / {metrics['aspectos']['micro_recall']}")
    print(f"  evidência literal:    {metrics['evidencia']['literal_no_texto']:.1%}")
    print(f"  JSON inválido:        {n_errors}")

    if save:
        save_section("llm_extracao", metrics)
        _save_side_by_side(rows)
    return metrics


def _save_side_by_side(rows):
    """CSV com gold e LLM lado a lado, para análise de erros."""
    out = []
    for g, p in rows:
        out.append({
            "review_id": g.name,
            "texto": g["review_text"],
            "gold_sent": g["sentiment"], "llm_sent": p["sentiment"],
            "gold_cats": g["categories"], "llm_cats": "|".join(p["categories"]),
            "gold_sev": g["severity"], "llm_sev": p["severity"],
            "llm_problem": p.get("problem"),
        })
    pd.DataFrame(out).to_csv(ARTIFACTS / "gold_llm_vs_gold.csv", index=False)

