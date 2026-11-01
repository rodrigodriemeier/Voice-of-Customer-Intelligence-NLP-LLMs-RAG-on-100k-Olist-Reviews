"""Gera reports/resultados.md a partir de reports/metrics.json e atualiza a
tabela de resultados do README (entre os marcadores <!-- resultados -->).

Mantém a tabela do README sincronizada com as métricas a cada execução.

Rodar:  python -m src.report
"""
import json
import re

from src.config import CATEGORIES, REPORTS, ROOT
from src.metrics_store import load_metrics

README = ROOT / "README.md"
START, END = "<!-- resultados:inicio -->", "<!-- resultados:fim -->"


# O README e o reports/resultados.md são em inglês; os rótulos internos
# (metrics.json, nomes dos métodos) continuam em português e são traduzidos aqui.
METHOD_EN = {
    "Só a nota": "Star rating only",
    "Regras": "Keyword rules",
    "TF-IDF + LogReg": "TF-IDF + LogReg",
    "LLM": "LLM",
    "TF-IDF (palavra-chave)": "TF-IDF (keyword)",
    "LSA (TF-IDF + SVD)": "LSA (TF-IDF + SVD)",
    "Aleatório (referência)": "Random (baseline)",
}
CATEGORY_EN = {
    "delivery": "Delivery",
    "product_quality": "Product quality / damaged",
    "wrong_product": "Wrong product / not as listed",
    "missing_items": "Incomplete order / missing item",
    "customer_service": "Customer service / after-sales",
}


def f2(x):
    if x is None:
        return "—"
    return f"{x:.2f}"


def num(n):
    return f"{n:,}"


def method(name):
    return METHOD_EN.get(name, name)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join([":--"] + ["--:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main_table(m):
    clf = m.get("classificador", {}).get("gold")
    llm = m.get("llm_extracao")
    if not clf:
        return "_Run `python run_pipeline.py` to generate the metrics._"
    L = (lambda k: f2(k(llm))) if llm else (lambda k: "—")
    rows = [
        ["Sentiment", f2(clf["sentimento"]["baseline_estrelas_macro_f1"]), "—",
         f2(clf["sentimento"]["tfidf_logreg_macro_f1"]), L(lambda x: x["sentimento_macro_f1"])],
        ["Complaint category", "—", f2(clf["categorias"]["baseline_regras"]["macro_f1"]),
         f2(clf["categorias"]["tfidf_logreg"]["macro_f1"]), L(lambda x: x["categorias"]["macro_f1"])],
        ["Severity", "—", "—", f2(clf["severidade"]["regras_macro_f1"]), L(lambda x: x["severidade_macro_f1"])],
    ]
    return table(["Macro-F1 on gold (n=400)", "Star rating only", "Keyword rules", "TF-IDF + LogReg", "LLM"], rows)


def per_category_table(m):
    clf = m["classificador"]["gold"]["categorias"]
    llm = m.get("llm_extracao")
    rows = []
    for cat in CATEGORIES:
        t = clf["tfidf_logreg"]["por_categoria"][cat]
        r = clf["baseline_regras"]["por_categoria"][cat]
        row = [CATEGORY_EN[cat], int(t["support"]), f2(r["f1-score"]), f2(t["precision"]), f2(t["recall"]), f2(t["f1-score"])]
        if llm:
            lr = llm["categorias"]["por_rotulo"][cat]
            row += [f2(lr["precision"]), f2(lr["recall"]), f2(lr["f1"])]
        rows.append(row)
    header = ["Category", "n", "Rules F1", "TF-IDF P", "TF-IDF R", "TF-IDF F1"]
    if llm:
        header += ["LLM P", "LLM R", "LLM F1"]
    return table(header, rows)


def search_table(m):
    b = m.get("busca")
    if not b:
        return "_Not run yet._"
    rows = [[method(name)] + [f2(v["media"][k]) for k in ("recall@20", "recall@50", "recall@100", "p@10")]
            for name, v in b["resultados"].items()]
    return table(["Method", "Recall@20", "Recall@50", "Recall@100", "P@10"], rows)


def llm_fields(m):
    llm = m.get("llm_extracao")
    if not llm:
        return "_Not run yet (requires an API key): `python -m src.llm_extract --gold`._"
    a = llm["aspectos"]
    rows = [
        ["Sentiment (macro-F1)", f2(llm["sentimento_macro_f1"])],
        ["Categories (micro P / R)", f"{f2(llm['categorias']['micro_precision'])} / {f2(llm['categorias']['micro_recall'])}"],
        ["Severity (macro-F1)", f2(llm["severidade_macro_f1"])],
        ["Aspects (micro P / R)", f"{f2(a['micro_precision'])} / {f2(a['micro_recall'])}"],
        ["Evidence copied verbatim from the review", f"{llm['evidencia']['literal_no_texto']:.1%}"],
        ["Evidence vs. annotation (token overlap)", f2(llm["evidencia"]["sobreposicao_tokens_media"])],
        ["Detects that there is a problem (P / R)", f"{f2(llm['problema_detectado']['precision'])} / {f2(llm['problema_detectado']['recall'])}"],
        ["Responses without valid JSON", f"{llm['respostas_sem_json_valido']} of {llm['n']}"],
    ]
    return f"Model: `{llm['modelo']}`\n\n" + table(["Field", "Result"], rows)


def rag_block(m):
    r = m.get("rag")
    if not r:
        return "_Not run yet (requires an API key): `python -m src.eval_rag`._"
    rows = [
        ["Faithfulness (claims supported by the cited reviews)", f2(r["faithfulness_media"])],
        ["Context relevance (retrieved reviews related to the question)", f2(r["relevancia_contexto_media"])],
        ["Answer relevance (1 to 5)", f2(r["relevancia_resposta_media"])],
        ["Answer segments with a citation", f2(r["pct_unidades_com_citacao_media"])],
        ["Answers with a number not computed in Python", f"{r['respostas_com_numero_inventado']} of {r['n_perguntas']}"],
    ]
    return (f"{r['n_perguntas']} questions · generator `{r['gerador']}` · judge `{r['juiz']}` · "
            f"embeddings `{r['embeddings']['model']}`\n\n" + table(["Metric", "Result"], rows))


def gold_review_block(m):
    r = m.get("revisao_gold")
    if not r or not r.get("n_revisadas"):
        return ("**Human review of the gold set:** not done yet. The current labels come from the first, "
                "LLM-assisted pass. See `docs/anotacao.md`.")
    lines = [f"**Human review of the gold set:** {r['n_revisadas']} of {r['n_total']} reviewed, "
             f"{r['n_revisadas_cegas']} in blind mode (without seeing the first-pass labels)."]
    a = r.get("concordancia_cega")
    if a:
        lines += ["", table(["Agreement: blind review vs. first pass", "Result"], [
            ["Sentiment (Cohen's kappa)", f2(a["sentimento"]["kappa"])],
            ["Severity (weighted kappa)", f2(a["severidade"]["kappa_ponderado"])],
            ["Categories: identical set", f"{a['categorias']['conjunto_identico']:.0%}"],
            ["Rows with any change", f"{a['linhas_com_alguma_mudanca']} of {a['n']}"],
        ])]
    sub = r.get("metricas_so_revisadas")
    if sub:
        rows = [[method(name), f2(v.get("sentimento")), f2(v.get("categorias")), f2(v.get("severidade"))]
                for name, v in sub.items()]
        lines += ["", f"Macro-F1 on the {r['n_revisadas']} human-reviewed rows only:", "",
                  table(["Method", "Sentiment", "Categories", "Severity"], rows)]
    return "\n".join(lines)


def index_line():
    path = ROOT / "artifacts" / "index" / "embedder.json"
    if not path.exists():
        return ""
    info = json.loads(path.read_text(encoding="utf-8"))
    if info["backend"] == "lsa":
        return ("**Search index shipped in the repository:** LSA (TF-IDF + SVD, "
                f"{info['dim']} dimensions), the offline fallback. To switch to Sentence Transformers: "
                "`python run_pipeline.py --from index`.")
    return f"**Search index shipped in the repository:** Sentence Transformers (`{info['model']}`)."


def build_markdown(m):
    h = m.get("classificador", {}).get("holdout", {})
    parts = [
        "# Results",
        "",
        "Generated by `python -m src.report` from `reports/metrics.json`. Do not edit by hand.",
        "",
        "## Classification",
        "",
        main_table(m),
        "",
        "### Per category",
        "",
        per_category_table(m) if m.get("classificador") else "",
        "",
    ]
    if h:
        parts += [
            f"Random holdout ({num(h['n'])} reviews, label taken from the star rating): sentiment macro-F1 "
            f"{f2(h['sentimento_vs_nota_macro_f1'])}; per-class F1: negative {f2(h['por_classe']['negative'])}, "
            f"neutral {f2(h['por_classe']['neutral'])}, positive {f2(h['por_classe']['positive'])}.",
            "",
        ]
    parts += ["## Reference set", "", gold_review_block(m), "",
              "## LLM extraction, field by field", "", llm_fields(m), "",
              "## Semantic search", "", search_table(m), "", index_line(), "",
              "## RAG", "", rag_block(m), ""]
    return "\n".join(parts)


def readme_block(m):
    return "\n".join([
        START,
        main_table(m),
        "",
        gold_review_block(m),
        "",
        "**Search** (15 queries, relevance given by the annotated category):",
        "",
        search_table(m),
        "",
        index_line(),
        "",
        "**RAG:**",
        "",
        rag_block(m),
        END,
    ])


def main():
    m = load_metrics()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "resultados.md").write_text(build_markdown(m), encoding="utf-8")
    print("reports/resultados.md updated")

    if README.exists():
        text = README.read_text(encoding="utf-8")
        if START in text and END in text:
            new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: readme_block(m), text, flags=re.S)
            README.write_text(new, encoding="utf-8")
            print("README results table updated")


if __name__ == "__main__":
    main()
