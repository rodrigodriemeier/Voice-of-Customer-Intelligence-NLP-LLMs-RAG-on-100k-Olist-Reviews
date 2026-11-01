import io

import pandas as pd
import streamlit as st

from app.common import br, classifier, llm_quota, llm_status_caption, metric
from src.config import CATEGORIES

st.title("Classificar uma avaliação")
st.caption("Cole o texto de uma avaliação. O classificador TF-IDF responde na hora; "
           "se houver LLM configurado, dá para comparar com a extração estruturada.")

model = classifier()

EXEMPLOS = [
    "Meu pedido chegou três dias atrasado e o produto veio danificado.",
    "Comprei duas unidades e só veio uma, já mandei e-mail e ninguém responde.",
    "Chegou antes do prazo, muito bem embalado. Recomendo!",
    "O tênis veio num número diferente do que pedi, mas a troca foi rápida.",
]

if "texto" not in st.session_state:
    st.session_state.texto = EXEMPLOS[0]

cols = st.columns(len(EXEMPLOS))
for c, ex in zip(cols, EXEMPLOS):
    if c.button(ex[:38] + "...", width="stretch", help=ex):
        st.session_state.texto = ex

text = st.text_area("Avaliação", key="texto", height=110, max_chars=1000)
client = llm_status_caption()
ask_llm = st.button("Comparar com o LLM", icon=":material/compare_arrows:", disabled=client is None or not text.strip())

if ask_llm:
    ok, msg = llm_quota()
    if not ok:
        st.warning(msg)
    else:
        from src.llm import LLMError
        from src.llm_extract import extract
        with st.spinner("Chamando o LLM..."):
            try:
                st.session_state.llm_res = {"text": text, "res": extract(text, client)}
            except LLMError:
                st.warning("O LLM não respondeu agora (crédito esgotado ou chave inválida).")

# resultado guardado na sessão: reexecuções da página não fazem nova chamada
stored = st.session_state.get("llm_res")
res = stored["res"] if stored and stored["text"] == text else None

if text.strip():
    out = model.predict(text)

    left, right = st.columns(2) if res else (st.container(), None)
    with left:
        st.markdown("**TF-IDF + regressão logística**")
        a, b, c = st.columns(3)
        a.metric("Sentimento", out["sentiment"])
        b.metric("Severidade", out["severity"])
        c.metric("Categorias", len(out["categories"]))
        st.json({k: out[k] for k in ("sentiment", "categories", "severity")})
        probs = pd.DataFrame(
            {"categoria": [CATEGORIES[c] for c in CATEGORIES], "probabilidade": [out["confidence"][c] for c in CATEGORIES]}
        ).sort_values("probabilidade", ascending=False)
        st.dataframe(
            probs, hide_index=True, width="stretch",
            column_config={"probabilidade": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
        )
        sev_f1 = metric("classificador", "gold", "severidade", "regras_macro_f1")
        st.caption(f"A severidade aqui vem de regras simples em cima das categorias (macro-F1 {br(sev_f1)} no gold). "
                   "É o ponto mais fraco do classificador clássico.")

    if res:
        with right:
            st.markdown(f"**LLM** (`{client.model}`)")
            if "error" in res:
                st.error(res["error"])
                st.code(res.get("raw", ""))
            else:
                a, b, c = st.columns(3)
                a.metric("Sentimento", res["sentiment"])
                b.metric("Severidade", res["severity"])
                c.metric("Categorias", len(res["categories"]))
                shown = {k: res[k] for k in ("sentiment", "categories", "severity", "aspects", "problem")}
                st.json(shown)
                bad = [x["evidence"] for x in res["aspects"] if not x["grounded"]]
                if bad:
                    st.warning("Evidência que não aparece literalmente no texto: " + "; ".join(bad))
                if res["issues"]:
                    st.caption("Campos corrigidos na validação: " + "; ".join(res["issues"]))

st.divider()
st.subheader("Várias de uma vez")
st.caption("Envie um CSV com uma coluna chamada `texto`. Volta o mesmo arquivo com as colunas do classificador.")
up = st.file_uploader("CSV", type="csv", label_visibility="collapsed")
if up is not None:
    df = pd.read_csv(up)
    if "texto" not in df.columns:
        st.error("Não achei a coluna `texto`.")
    else:
        if len(df) > 20_000:
            st.warning("Classifiquei só as primeiras 20.000 linhas.")
            df = df.head(20_000)
        pred = model.predict_frame(df["texto"].fillna("").astype(str).tolist())
        df["sentimento"] = pred["sentiment"]
        df["categorias"] = pred["categories"].map("|".join)
        df["severidade"] = pred["severity"]
        st.dataframe(df.head(50), width="stretch")
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Baixar CSV classificado", buf.getvalue(), file_name="avaliacoes_classificadas.csv",
                           mime="text/csv", icon=":material/download:")
