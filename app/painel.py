import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.common import CATEGORY_ORDER, analytics, br, category_colors, metric, palette, style
from src.config import CATEGORIES

a = analytics()
macros, states, (dmin, dmax) = a.options()

st.title("Painel das avaliações")
st.caption(
    f"{dmin:%d/%m/%Y} a {dmax:%d/%m/%Y} · todos os números são calculados em SQL sobre a base, "
    "nada aqui é estimado por LLM."
)

# --- filtros, todos numa linha -------------------------------------------------
c1, c2, c3 = st.columns([1.2, 2, 1.4])
period = c1.date_input("Período", value=(dmin.date(), dmax.date()), min_value=dmin.date(),
                       max_value=dmax.date(), format="DD/MM/YYYY")
sel_macro = c2.multiselect("Categoria de produto", macros, placeholder="Todas")
sel_states = c3.multiselect("Estado do cliente", states, placeholder="Todos")

filters = {"macro": sel_macro, "states": sel_states}
if isinstance(period, (list, tuple)) and len(period) == 2:
    filters["date_from"], filters["date_to"] = period

k = a.kpis(filters)
if not k["avaliacoes"]:
    st.warning("Nenhuma avaliação com esses filtros.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Avaliações", f"{int(k['avaliacoes']):,}".replace(",", "."))
m2.metric("Nota média", f"{k['nota_media']:.2f}".replace(".", ","))
m3.metric("Negativas (nota 1 ou 2)", f"{k['pct_negativas']:.1%}".replace(".", ","))
m4.metric("Pedidos entregues com atraso", f"{k['pct_atraso']:.1%}".replace(".", ","))

st.divider()

# --- negativas por categoria + causas ------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Onde a insatisfação é maior")
    level = st.segmented_control("Agrupar por", ["Macrocategoria", "Categoria"], default="Macrocategoria",
                                 label_visibility="collapsed")
    col = "macro_category" if level != "Categoria" else "product_category"
    neg = a.negative_rate_by(filters, level=col, min_reviews=300 if col == "macro_category" else 400)
    neg = neg[neg["grupo"] != "Sem categoria"].head(15).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=neg["pct_negativas"], y=neg["grupo"], orientation="h", marker_color=palette()[0],
        customdata=neg["avaliacoes"],
        hovertemplate="%{y}<br>%{x:.1%} negativas<br>%{customdata:,} avaliações<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, height=430), width="stretch")
    st.caption("% de avaliações com nota 1 ou 2. Só grupos com volume suficiente para a taxa não ficar instável.")

with right:
    st.subheader("Principais causas de insatisfação")
    causes = a.complaint_causes(filters)
    colors = category_colors()
    causes = causes.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=causes["pct"], y=causes["categoria"], orientation="h",
        marker_color=[colors.get(c, palette()[0]) for c in causes["cat"]],
        customdata=causes["n"],
        hovertemplate="%{y}<br>%{x:.1%} das avaliações com problema<br>%{customdata:,} avaliações<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(bargap=0.5)
    st.plotly_chart(style(fig, height=430), width="stretch")
    st.caption(
        "Entre avaliações com texto e nota até 3. A categoria é prevista pelo classificador TF-IDF "
        f"(macro-F1 {br(metric('classificador', 'gold', 'categorias', 'tfidf_logreg', 'macro_f1'))} no conjunto "
        "anotado), então trate como estimativa. Uma avaliação pode ter mais de uma."
    )

st.divider()

# --- atraso x nota --------------------------------------------------------------
st.subheader("Atraso na entrega x avaliação negativa")
delay = a.delay_vs_negative(filters)
fig = go.Figure(go.Bar(
    x=delay["faixa"].astype(str), y=delay["pct_negativas"], marker_color=palette()[0],
    customdata=delay[["avaliacoes", "nota_media"]],
    hovertemplate="%{x}<br>%{y:.1%} negativas<br>nota média %{customdata[1]:.2f}"
                  "<br>%{customdata[0]:,} avaliações<extra></extra>",
))
fig.update_yaxes(tickformat=".0%")
fig.update_layout(bargap=0.55)
st.plotly_chart(style(fig, height=340), width="stretch")

on_time = delay[delay["faixa"].astype(str).str.contains("adiantado")]
late = delay[delay["faixa"].astype(str).isin(["8-14 dias atrasado", "15+ dias atrasado"])]
short = delay[delay["faixa"].astype(str) == "1-3 dias atrasado"]
if len(on_time) and len(late) and len(short):
    r_on = (on_time["pct_negativas"] * on_time["avaliacoes"]).sum() / on_time["avaliacoes"].sum()
    r_late = (late["pct_negativas"] * late["avaliacoes"]).sum() / late["avaliacoes"].sum()
    r_short = short["pct_negativas"].iloc[0]
    st.markdown(
        f"Quando o pedido chega no prazo, **{r_on:.0%}** das avaliações são negativas. "
        f"Com 1 a 3 dias de atraso já são **{r_short:.0%}**, e com mais de uma semana, **{r_late:.0%}**. "
        "É correlação, não prova de causa (pedido atrasado também tende a ter outros problemas junto)."
    )

st.divider()

# --- evolução no tempo ------------------------------------------------------------
st.subheader("Evolução mensal")
base, cats = a.monthly(filters)

t1, t2 = st.columns(2)
with t1:
    st.markdown("**% de avaliações negativas**")
    fig = go.Figure(go.Scatter(
        x=base["mes"], y=base["pct_negativas"], mode="lines+markers", line=dict(width=2, color=palette()[0]),
        marker=dict(size=8), customdata=base["avaliacoes"],
        hovertemplate="%{x|%b/%Y}<br>%{y:.1%} negativas<br>%{customdata:,} avaliações<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(hovermode="x")
    st.plotly_chart(style(fig, height=400), width="stretch")

with t2:
    st.markdown("**% das avaliações com texto, por tipo de reclamação**")
    colors = category_colors()
    order = [CATEGORIES[c].split(" (")[0] for c in CATEGORY_ORDER]
    fig = px.line(
        cats, x="mes", y="pct", color="categoria", markers=True,
        category_orders={"categoria": order},
        color_discrete_map={CATEGORIES[c].split(" (")[0]: colors[c] for c in CATEGORY_ORDER},
    )
    fig.update_traces(line_width=2, marker_size=7,
                      hovertemplate="%{fullData.name}<br>%{x|%b/%Y}: %{y:.1%}<extra></extra>")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    fig = style(fig, height=400)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, x=0))
    st.plotly_chart(fig, width="stretch")

with st.expander("Ver os dados destes gráficos em tabela"):
    tab = cats.pivot(index="mes", columns="categoria", values="pct").reindex(columns=order)
    tab.insert(0, "% negativas", base.set_index("mes")["pct_negativas"])
    tab.index = tab.index.strftime("%m/%Y")
    st.dataframe(tab.style.format("{:.1%}"), width="stretch")

st.divider()

# --- exemplos ---------------------------------------------------------------------
st.subheader("Exemplos de reclamações")
chosen = st.selectbox("Tipo de reclamação", CATEGORY_ORDER, format_func=lambda c: CATEGORIES[c])
ex = a.examples(filters, chosen, n=6)
if ex.empty:
    st.info("Nenhum exemplo com esses filtros.")
for row in ex.itertuples():
    st.markdown(f"> {row.review_text}  \n<small>nota {row.review_score} · {row.product_category}</small>",
                unsafe_allow_html=True)
if st.button("Sortear outros", icon=":material/refresh:"):
    st.rerun()
