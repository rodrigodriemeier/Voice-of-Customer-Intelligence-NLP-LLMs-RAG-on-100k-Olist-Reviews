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


def f2(x):
    if x is None:
        return "—"
    return f"{x:.2f}".replace(".", ",")


def num(n):
    return f"{n:,}".replace(",", ".")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join([":--"] + ["--:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main_table(m):
    clf = m.get("classificador", {}).get("gold")
    llm = m.get("llm_extracao")
    if not clf:
        return "_Rode `python run_pipeline.py` para gerar as métricas._"
    L = (lambda k: f2(k(llm))) if llm else (lambda k: "—")
    rows = [
        ["Sentimento", f2(clf["sentimento"]["baseline_estrelas_macro_f1"]), "—",
         f2(clf["sentimento"]["tfidf_logreg_macro_f1"]), L(lambda x: x["sentimento_macro_f1"])],
        ["Categoria da reclamação", "—", f2(clf["categorias"]["baseline_regras"]["macro_f1"]),
         f2(clf["categorias"]["tfidf_logreg"]["macro_f1"]), L(lambda x: x["categorias"]["macro_f1"])],
        ["Severidade", "—", "—", f2(clf["severidade"]["regras_macro_f1"]), L(lambda x: x["severidade_macro_f1"])],
    ]
    return table(["Macro-F1 no gold (n=400)", "Só a nota", "Regras", "TF-IDF + LogReg", "LLM"], rows)


def per_category_table(m):
    clf = m["classificador"]["gold"]["categorias"]
    llm = m.get("llm_extracao")
    rows = []
    for cat, label in CATEGORIES.items():
        t = clf["tfidf_logreg"]["por_categoria"][cat]
        r = clf["baseline_regras"]["por_categoria"][cat]
        row = [label.split(" (")[0], int(t["support"]), f2(r["f1-score"]), f2(t["precision"]), f2(t["recall"]), f2(t["f1-score"])]
        if llm:
            lr = llm["categorias"]["por_rotulo"][cat]
            row += [f2(lr["precision"]), f2(lr["recall"]), f2(lr["f1"])]
        rows.append(row)
    header = ["Categoria", "n", "Regras F1", "TF-IDF P", "TF-IDF R", "TF-IDF F1"]
    if llm:
        header += ["LLM P", "LLM R", "LLM F1"]
    return table(header, rows)


def search_table(m):
    b = m.get("busca")
    if not b:
        return "_Ainda não rodado._"
    rows = [[name] + [f2(v["media"][k]) for k in ("recall@20", "recall@50", "recall@100", "p@10")]
            for name, v in b["resultados"].items()]
    return table(["Método", "Recall@20", "Recall@50", "Recall@100", "P@10"], rows)


def llm_fields(m):
    llm = m.get("llm_extracao")
    if not llm:
        return "_Ainda não rodado (precisa de chave de API): `python -m src.llm_extract --gold`._"
    a = llm["aspectos"]
    rows = [
        ["Sentimento (macro-F1)", f2(llm["sentimento_macro_f1"])],
        ["Categorias (micro P / R)", f"{f2(llm['categorias']['micro_precision'])} / {f2(llm['categorias']['micro_recall'])}"],
        ["Severidade (macro-F1)", f2(llm["severidade_macro_f1"])],
        ["Aspectos (micro P / R)", f"{f2(a['micro_precision'])} / {f2(a['micro_recall'])}"],
        ["Evidência copiada literalmente do texto", f"{llm['evidencia']['literal_no_texto']:.0%}"],
        ["Evidência x anotação (sobreposição de tokens)", f2(llm["evidencia"]["sobreposicao_tokens_media"])],
        ["Detectou que há problema (P / R)", f"{f2(llm['problema_detectado']['precision'])} / {f2(llm['problema_detectado']['recall'])}"],
        ["Respostas sem JSON válido", f"{llm['respostas_sem_json_valido']} de {llm['n']}"],
    ]
    return f"Modelo: `{llm['modelo']}`\n\n" + table(["Campo", "Resultado"], rows)


def rag_block(m):
    r = m.get("rag")
    if not r:
        return "_Ainda não rodado (precisa de chave de API): `python -m src.eval_rag`._"
    rows = [
        ["Faithfulness (afirmações sustentadas pelas avaliações citadas)", f2(r["faithfulness_media"])],
        ["Relevância do contexto (avaliações recuperadas que têm a ver)", f2(r["relevancia_contexto_media"])],
        ["Relevância da resposta (1 a 5)", f2(r["relevancia_resposta_media"])],
        ["Trechos da resposta com citação", f2(r["pct_unidades_com_citacao_media"])],
        ["Respostas com número que não veio do Python", f"{r['respostas_com_numero_inventado']} de {r['n_perguntas']}"],
    ]
    return (f"{r['n_perguntas']} perguntas · gerador `{r['gerador']}` · juiz `{r['juiz']}` · "
            f"embeddings `{r['embeddings']['model']}`\n\n" + table(["Métrica", "Resultado"], rows))


def gold_review_block(m):
    r = m.get("revisao_gold")
    if not r or not r.get("n_revisadas"):
        return ("**Revisão humana do gold:** ainda não feita. Os rótulos atuais são da primeira passada "
                "(assistida por LLM). Ver `docs/anotacao.md`.")
    lines = [f"**Revisão humana do gold:** {r['n_revisadas']} de {r['n_total']} revisadas, "
             f"{r['n_revisadas_cegas']} em modo cego (sem ver os rótulos da primeira passada)."]
    a = r.get("concordancia_cega")
    if a:
        lines += ["", table(["Concordância revisão cega x primeira passada", "Resultado"], [
            ["Sentimento (kappa)", f2(a["sentimento"]["kappa"])],
            ["Severidade (kappa ponderado)", f2(a["severidade"]["kappa_ponderado"])],
            ["Categorias: conjunto idêntico", f"{a['categorias']['conjunto_identico']:.0%}"],
            ["Linhas com alguma mudança", f"{a['linhas_com_alguma_mudanca']} de {a['n']}"],
        ])]
    sub = r.get("metricas_so_revisadas")
    if sub:
        rows = [[name, f2(v.get("sentimento")), f2(v.get("categorias")), f2(v.get("severidade"))]
                for name, v in sub.items()]
        lines += ["", f"Macro-F1 só nas {r['n_revisadas']} revisadas:", "",
                  table(["Método", "Sentimento", "Categorias", "Severidade"], rows)]
    return "\n".join(lines)


def index_line():
    path = ROOT / "artifacts" / "index" / "embedder.json"
    if not path.exists():
        return ""
    info = json.loads(path.read_text(encoding="utf-8"))
    if info["backend"] == "lsa":
        return ("**Índice de busca incluído no repositório:** LSA (TF-IDF + SVD, "
                f"{info['dim']} dimensões), o plano B. Para trocar por Sentence Transformers: "
                "`python run_pipeline.py --from index`.")
    return f"**Índice de busca incluído no repositório:** Sentence Transformers (`{info['model']}`)."


def build_markdown(m):
    h = m.get("classificador", {}).get("holdout", {})
    parts = [
        "# Resultados",
        "",
        "Arquivo gerado por `python -m src.report` a partir de `reports/metrics.json`. Não edite à mão.",
        "",
        "## Classificação",
        "",
        main_table(m),
        "",
        "### Por categoria",
        "",
        per_category_table(m) if m.get("classificador") else "",
        "",
    ]
    if h:
        parts += [
            f"Holdout aleatório ({num(h['n'])} avaliações, rótulo vindo da nota): macro-F1 de sentimento "
            f"{f2(h['sentimento_vs_nota_macro_f1'])}; F1 por classe: negativo {f2(h['por_classe']['negative'])}, "
            f"neutro {f2(h['por_classe']['neutral'])}, positivo {f2(h['por_classe']['positive'])}.",
            "",
        ]
    parts += ["## Conjunto de referência", "", gold_review_block(m), "",
              "## Extração com LLM, campo a campo", "", llm_fields(m), "",
              "## Busca semântica", "", search_table(m), "", index_line(), "",
              "## RAG", "", rag_block(m), ""]
    return "\n".join(parts)


def readme_block(m):
    return "\n".join([
        START,
        main_table(m),
        "",
        gold_review_block(m),
        "",
        "**Busca** (15 consultas, relevância pela categoria anotada):",
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
    print("reports/resultados.md atualizado")

    if README.exists():
        text = README.read_text(encoding="utf-8")
        if START in text and END in text:
            new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: readme_block(m), text, flags=re.S)
            README.write_text(new, encoding="utf-8")
            print("tabela do README atualizada")


if __name__ == "__main__":
    main()
