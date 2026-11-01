import pandas as pd
import streamlit as st

from src.config import CATEGORIES
from src.metrics_store import load_metrics

st.title("Como os modelos se saem")
st.caption("Tudo medido no conjunto de referência de 400 avaliações anotadas (`data/annotations/gold_400.csv`). "
           "A amostra puxa mais avaliações ruins do que a base real, então as taxas não representam a distribuição "
           "real. Elas servem para comparar os métodos entre si.")

m = load_metrics()
if not m:
    st.warning("Ainda não há métricas. Rode `python run_pipeline.py`.")
    st.stop()


def pct(x):
    if x is None or (isinstance(x, float) and x != x):  # None ou NaN
        return "—"
    return f"{x:.2f}".replace(".", ",")


def num(n):
    return f"{n:,}".replace(",", ".")


# --- classificação -----------------------------------------------------------
st.header("Classificação: TF-IDF x LLM")
clf = m.get("classificador", {}).get("gold")
llm = m.get("llm_extracao")
if clf:
    rows = [
        ["Sentimento (macro-F1)", clf["sentimento"]["baseline_estrelas_macro_f1"], None,
         clf["sentimento"]["tfidf_logreg_macro_f1"], llm and llm["sentimento_macro_f1"]],
        ["Categorias (macro-F1)", None, clf["categorias"]["baseline_regras"]["macro_f1"],
         clf["categorias"]["tfidf_logreg"]["macro_f1"], llm and llm["categorias"]["macro_f1"]],
        ["Severidade (macro-F1)", None, None, clf["severidade"]["regras_macro_f1"], llm and llm["severidade_macro_f1"]],
    ]
    df = pd.DataFrame(rows, columns=["", "Só a nota (estrelas)", "Regras", "TF-IDF + LogReg", "LLM"])
    st.dataframe(df.set_index("").map(pct), width="stretch")
    if not llm:
        st.caption("A coluna do LLM aparece depois de rodar `python -m src.llm_extract --gold`.")
    else:
        st.caption(f"LLM: `{llm['modelo']}` · {llm['respostas_sem_json_valido']} respostas sem JSON válido em {llm['n']}.")

    st.subheader("Por categoria de reclamação")
    per = []
    for cat, label in CATEGORIES.items():
        t = clf["categorias"]["tfidf_logreg"]["por_categoria"][cat]
        r = clf["categorias"]["baseline_regras"]["por_categoria"][cat]
        row = {"categoria": label, "suporte": int(t["support"]),
               "regras F1": r["f1-score"], "TF-IDF P": t["precision"], "TF-IDF R": t["recall"], "TF-IDF F1": t["f1-score"]}
        if llm:
            lr = llm["categorias"]["por_rotulo"][cat]
            row.update({"LLM P": lr["precision"], "LLM R": lr["recall"], "LLM F1": lr["f1"]})
        per.append(row)
    per = pd.DataFrame(per).set_index("categoria")
    st.dataframe(per.drop(columns="suporte").map(pct).assign(suporte=per["suporte"]), width="stretch")

    h = m["classificador"]["holdout"]
    st.caption(
        f"No holdout aleatório ({num(h['n'])} avaliações, rótulo = nota), o sentimento do TF-IDF tem macro-F1 "
        f"{pct(h['sentimento_vs_nota_macro_f1'])}. A classe neutra é a pior "
        f"(F1 {pct(h['por_classe']['neutral'])}): no gold, mais da metade das avaliações nota 3 são reclamação "
        "quando se lê o texto, e o rótulo 'neutro' tirado da nota erra em 4 de cada 5 casos."
    )

if llm:
    st.subheader("Extração via LLM, campo a campo")
    a = llm["aspectos"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aspectos: precision", pct(a["micro_precision"]))
    c2.metric("Aspectos: recall", pct(a["micro_recall"]))
    c3.metric("Evidência literal no texto", f"{llm['evidencia']['literal_no_texto']:.0%}")
    c4.metric("Detecta que há problema (F1)", pct(llm["problema_detectado"]["f1"]))

# --- revisão do gold ------------------------------------------------------------
st.header("Revisão do conjunto de referência")
rg = m.get("revisao_gold")
if not rg or not rg.get("n_revisadas"):
    st.warning("Nenhuma avaliação do gold foi revisada ainda: os rótulos são da primeira passada, assistida por "
               "LLM. Use a página “Revisar anotações” (modo cego) e depois `python -m src.eval_gold`.")
else:
    st.progress(rg["n_revisadas"] / rg["n_total"],
                text=f"{rg['n_revisadas']} de {rg['n_total']} revisadas · {rg['n_revisadas_cegas']} em modo cego")
    a = rg.get("concordancia_cega")
    if a:
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentimento (kappa)", pct(a["sentimento"]["kappa"]))
        c2.metric("Severidade (kappa ponderado)", pct(a["severidade"]["kappa_ponderado"]))
        c3.metric("Categorias idênticas", f"{a['categorias']['conjunto_identico']:.0%}")
        st.caption("Concordância entre a revisão cega e a primeira passada. Kappa acima de ~0,8 costuma ser "
                   "lido como concordância forte; abaixo de ~0,6, o rótulo precisa de critério mais claro.")
    sub = rg.get("metricas_so_revisadas")
    if sub:
        st.markdown(f"**Macro-F1 só nas {rg['n_revisadas']} revisadas**")
        st.dataframe(pd.DataFrame(sub).T.map(pct), width="stretch")

# --- busca ---------------------------------------------------------------------
st.header("Busca semântica")
busca = m.get("busca")
if busca:
    res = pd.DataFrame({k: v["media"] for k, v in busca["resultados"].items()}).T
    res.columns = ["Recall@20", "Recall@50", "Recall@100", "P@10"]
    st.dataframe(res.map(pct), width="stretch")
    st.caption(f"{busca['n_consultas']} consultas (3 por categoria, escritas de jeitos diferentes) buscando entre "
               f"as {busca['n_docs']} avaliações do gold. Relevante = anotada com a categoria da consulta.")
    if not any("Sentence" in k for k in busca["resultados"]):
        st.caption("Sentence Transformers não estava disponível quando a avaliação rodou. "
                   "Instale `sentence-transformers` e rode `python -m src.eval_search` para incluir.")

# --- RAG -------------------------------------------------------------------------
st.header("Respostas do RAG")
rag = m.get("rag")
if not rag:
    st.caption("Aparece depois de rodar `python -m src.eval_rag` (precisa de LLM).")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faithfulness", pct(rag["faithfulness_media"]))
    c2.metric("Relevância do contexto", pct(rag["relevancia_contexto_media"]))
    c3.metric("Relevância da resposta (1-5)", pct(rag["relevancia_resposta_media"]))
    c4.metric("Respostas com número inventado", f"{rag['respostas_com_numero_inventado']} de {rag['n_perguntas']}")
    st.dataframe(pd.DataFrame(rag["por_pergunta"]), width="stretch", hide_index=True)
    st.caption(f"Gerador `{rag['gerador']}`, juiz `{rag['juiz']}`, embeddings `{rag['embeddings']['model']}`.")
