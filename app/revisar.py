"""Tela para revisar o conjunto de referência.

Uso local: grava diretamente em data/annotations/gold_400.csv
(no Streamlit Cloud o disco é temporário).

- Ordem aleatória com semente fixa: revisando em sequência, as primeiras N
  linhas formam uma amostra aleatória do gold.
- Modo cego (padrão): os rótulos da primeira passada ficam ocultos. Com os
  rótulos visíveis, a revisão tende a confirmá-los e a concordância medida
  em src/eval_gold.py fica inflada.
"""
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ASPECTS, CATEGORIES, GOLD_FILE, SENTIMENTS, SEVERITIES
from src.gold import split_list

SEED = 2026

st.title("Revisar anotações")
st.caption(
    "Critério em `docs/anotacao.md`. A ordem é aleatória e fixa: revisando em sequência, as primeiras N já "
    "são uma amostra aleatória. Depois rode `python -m src.eval_gold` para medir a concordância."
)


def load():
    df = pd.read_csv(GOLD_FILE, keep_default_na=False, dtype=str)
    df["reviewed"] = df["reviewed"].str.lower().isin(["true", "1"])
    if "review_mode" not in df.columns:
        df["review_mode"] = ""
    return df


df = load()
order = np.random.default_rng(SEED).permutation(len(df))
done = int(df["reviewed"].sum())
blind_done = int((df["review_mode"] == "cega").sum())
st.progress(done / len(df), text=f"{done} de {len(df)} revisadas ({blind_done} em modo cego)")

c1, c2 = st.columns(2)
blind = c1.toggle("Modo cego (não mostra os rótulos da primeira passada)", value=True)
only_pending = c2.toggle("Mostrar só as que faltam", value=True)

pool = [i for i in order if not df.loc[i, "reviewed"]] if only_pending else list(order)
if not pool:
    st.success("Tudo revisado.")
    st.stop()

if "pos" not in st.session_state or st.session_state.pos >= len(pool):
    st.session_state.pos = 0
nav1, nav2, nav3 = st.columns([1, 1, 4])
if nav1.button("Anterior", icon=":material/arrow_back:", disabled=st.session_state.pos == 0):
    st.session_state.pos -= 1
    st.rerun()
if nav2.button("Pular", icon=":material/arrow_forward:"):
    st.session_state.pos = (st.session_state.pos + 1) % len(pool)
    st.rerun()

i = pool[st.session_state.pos]
row = df.loc[i]
nav3.caption(f"posição {int(np.where(order == i)[0][0]) + 1} na ordem · review_id `{row['review_id']}` · "
             f"{row['macro_category']}")

with st.container(border=True):
    st.markdown(f"#### {row['review_text']}")
    with st.popover("Ver nota em estrelas"):
        st.write(f"{row['review_score']} ★  (melhor olhar só depois de decidir o sentimento)")


def default(value, options):
    if blind:
        return None
    return options.index(value) if value in options else None


with st.form(f"f{i}_{blind}"):
    c1, c2 = st.columns(2)
    sent = c1.radio("Sentimento", SENTIMENTS, index=default(row["sentiment"], SENTIMENTS), horizontal=True)
    sev = c2.radio("Severidade", SEVERITIES, index=default(row["severity"], SEVERITIES), horizontal=True)
    cats = st.multiselect("Categorias de reclamação", list(CATEGORIES), format_func=lambda c: CATEGORIES[c],
                          default=[] if blind else split_list(row["categories"]), placeholder="Nenhuma")
    asp = st.multiselect("Aspectos mencionados", ASPECTS, placeholder="Nenhum",
                         default=[] if blind else [a for a in split_list(row["aspects"]) if a in ASPECTS])
    help_keep = "Em branco = mantém o da primeira passada." if blind else None
    ev = st.text_input("Evidência (trecho copiado do texto)", value="" if blind else row["evidence"], help=help_keep)
    prob = st.text_input("Problema (com suas palavras)", value="" if blind else row["problem"], help=help_keep)
    notes = st.text_input("Observações", value="" if blind else row["notes"], help=help_keep)
    ok = st.form_submit_button("Salvar e ir para a próxima", type="primary")

if ok:
    errors = []
    if sent is None or sev is None:
        errors.append("Escolha o sentimento e a severidade.")
    if cats and sev == "none":
        errors.append("Tem categoria de reclamação mas a severidade está 'none'.")
    if ev and ev not in row["review_text"]:
        errors.append("A evidência precisa ser um trecho copiado exatamente do texto.")
    has_problem = bool(cats) or sev not in (None, "none")
    if blind and has_problem and not prob and not row["problem"]:
        errors.append("Você marcou um problema; descreva em poucas palavras.")
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    if blind:
        ev = ev or row["evidence"]
        prob = prob or (row["problem"] if has_problem else "")
        notes = notes or row["notes"]

    df.loc[i, ["sentiment", "severity", "categories", "aspects", "evidence", "problem", "notes"]] = [
        sent, sev, "|".join(cats), "|".join(asp), ev, prob, notes
    ]
    df.loc[i, "reviewed"] = True
    df.loc[i, "review_mode"] = "cega" if blind else "com_rotulos"
    df.to_csv(GOLD_FILE, index=False)
    if not only_pending:
        st.session_state.pos += 1
    st.rerun()
