"""Ponto de entrada do app:  streamlit run streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="Avaliações Olist", page_icon=":material/reviews:", layout="wide")

pages = [
    st.Page("app/painel.py", title="Painel", icon=":material/monitoring:", default=True),
    st.Page("app/classificar.py", title="Classificar avaliação", icon=":material/label:"),
    st.Page("app/perguntar.py", title="Perguntar às avaliações", icon=":material/forum:"),
    st.Page("app/metricas.py", title="Como os modelos se saem", icon=":material/fact_check:"),
    st.Page("app/revisar.py", title="Revisar anotações", icon=":material/edit_note:"),
]

st.navigation(pages).run()
