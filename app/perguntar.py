import pandas as pd
import streamlit as st

from app.common import analytics, llm_quota, llm_status_caption, review_index
from src.llm import LLMError
from src.rag import Filters, answer, filters_from_question

st.title("Perguntar às avaliações")
st.caption(
    "A pergunta vira uma busca semântica nas ~42 mil avaliações com texto. As mais parecidas vão para o LLM, "
    "que responde citando cada uma como [R1], [R2]... Os números que aparecem na resposta são calculados em "
    "Python e passados prontos; o LLM é instruído a não inventar nenhum."
)

index = review_index()
macros, states, _ = analytics().options()
client = llm_status_caption()
if index.backend["backend"] == "lsa":
    st.info(
        "O índice de busca desta instância usa **LSA (TF-IDF + SVD)**, o plano B do projeto, e não Sentence "
        "Transformers. A busca funciona, mas pega menos sinônimos. Para usar embeddings de verdade: "
        "`python run_pipeline.py --from index` numa máquina com acesso ao Hugging Face.",
        icon=":material/info:",
    )
else:
    st.caption(f"Embeddings: `{index.backend['model']}`")

SUGESTOES = [
    "Quais são os principais problemas relacionados à entrega de produtos eletrônicos?",
    "O que os clientes reclamam quando o pedido chega incompleto?",
    "Quais problemas aparecem em produtos de beleza e perfumaria?",
    "Como os clientes reagem quando o pedido é cancelado por falta de estoque?",
    "O que os clientes elogiam na entrega quando chega antes do prazo?",
]

question = st.selectbox("Sugestões", SUGESTOES, index=None, placeholder="Escolha uma sugestão ou escreva abaixo")
question = st.text_input("Pergunta", value=question or "", max_chars=300,
                         placeholder="Ex.: o que os clientes dizem sobre embalagem?")

if question:
    auto = filters_from_question(question)
    with st.expander("Filtros (preenchidos a partir da pergunta, dá para mudar)", expanded=False):
        f1, f2, f3 = st.columns([2, 1, 1])
        sel_macro = f1.multiselect("Categoria de produto", macros, default=auto.macro_categories)
        only_bad = f2.checkbox("Só avaliações com nota ≤ 3", value=auto.only_problems)
        k = f3.slider("Avaliações recuperadas", 4, 15, 8)
        sel_states = st.multiselect("Estado", states)

    filters = Filters(macro_categories=sel_macro, only_problems=only_bad, states=sel_states)

    if st.button("Perguntar", type="primary", icon=":material/send:"):
        use = client
        if client is not None:
            ok, msg = llm_quota()
            if not ok:
                st.warning(msg + " Mostrando o resumo automático no lugar.")
                use = None
        with st.spinner("Buscando e escrevendo a resposta..."):
            try:
                res = answer(question, index, client=use, k=k, filters=filters)
            except LLMError:
                # crédito acabou, chave revogada etc.: o app continua, só sem o LLM
                st.warning("O LLM não respondeu agora (crédito esgotado ou chave inválida). "
                           "Mostrando o resumo automático no lugar.")
                res = answer(question, index, client=None, k=k, filters=filters)
        st.session_state.rag = res

res = st.session_state.get("rag")
if res and res["question"] == question:
    st.divider()
    if not res["used_llm"]:
        st.info("Resumo automático das avaliações recuperadas (sem LLM).")
    st.markdown(res["answer"])
    if res["invalid_citations"]:
        st.warning(f"A resposta citou avaliações que não existem na lista: {res['invalid_citations']}")

    meta = st.columns(3)
    meta[0].caption(f"Filtros: {res['filters'].describe()}")
    meta[1].caption(f"Texto usado na busca: “{res['search_query']}”")
    meta[2].caption(f"{len(res['cited'])} de {len(res['hits'])} avaliações citadas")

    with st.expander("Números passados ao LLM (calculados em Python)"):
        st.json(res["numbers"])

    st.subheader("Avaliações usadas como evidência")
    for i, row in enumerate(res["hits"].itertuples(), start=1):
        cited = i in res["cited"]
        date = pd.Timestamp(row.review_creation_date).strftime("%m/%Y")
        with st.container(border=True):
            head = f"**R{i}**" + ("  ·  citada" if cited else "")
            st.markdown(head)
            st.write(row.review_text)
            st.caption(f"nota {row.review_score} · {row.product_category} · {row.customer_state} · {date} · "
                       f"similaridade {row.similarity:.2f} · id `{row.review_id[:8]}`")
