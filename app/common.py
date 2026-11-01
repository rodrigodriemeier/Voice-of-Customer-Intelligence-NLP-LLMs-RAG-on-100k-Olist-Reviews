"""Coisas compartilhadas entre as páginas: carregamento com cache, cores, avisos."""
import streamlit as st

from src.config import ARTIFACTS, DATA_PROCESSED

# Paleta categórica validada para daltonismo (ordem fixa, não mudar a ordem:
# a separação entre cores vizinhas depende dela). Uma versão para cada tema.
_PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}

# cada categoria de reclamação tem sempre a mesma cor, em qualquer gráfico
CATEGORY_ORDER = ["delivery", "customer_service", "missing_items", "product_quality", "wrong_product"]


def theme():
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def palette():
    return _PALETTE.get(theme(), _PALETTE["light"])


def category_colors():
    pal = palette()
    return {cat: pal[i] for i, cat in enumerate(CATEGORY_ORDER)}


def style(fig, height=360):
    grid = "rgba(128,128,128,0.18)"
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        hoverlabel=dict(namelength=-1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        bargap=fig.layout.bargap if fig.layout.bargap is not None else 0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor=grid, title_text="")
    fig.update_yaxes(gridcolor=grid, gridwidth=1, zeroline=False, title_text="")
    return fig


def need(*paths, how="python run_pipeline.py"):
    missing = [p for p in paths if not p.exists()]
    if missing:
        st.error(
            "Faltam arquivos gerados pelo pipeline:\n\n"
            + "\n".join(f"- `{p.relative_to(p.parents[2])}`" for p in missing)
            + f"\n\nRode `{how}` na raiz do projeto e recarregue a página."
        )
        st.stop()


@st.cache_resource(show_spinner="Abrindo a base...")
def analytics():
    from src.analytics import REVIEWS, SCORED, Analytics
    need(REVIEWS, SCORED)
    return Analytics()


@st.cache_resource(show_spinner="Carregando o classificador...")
def classifier():
    from src.classifier import MODEL_FILE, ReviewClassifier
    need(MODEL_FILE)
    return ReviewClassifier.load()


@st.cache_resource(show_spinner="Carregando o índice de busca (na primeira vez pode baixar o modelo)...")
def review_index():
    from src.index import INDEX_DIR, ReviewIndex
    need(INDEX_DIR / "reviews.faiss", INDEX_DIR / "meta.parquet", how="python -m src.index")
    return ReviewIndex()


@st.cache_resource
def llm():
    from src.llm import get_client
    return get_client()


def llm_status_caption():
    from src.usage import SESSION_LIMIT
    client = llm()
    if client is None:
        st.caption(
            "LLM não configurado. Para ativar, crie um `.env` com `ANTHROPIC_API_KEY` "
            "(ou `OPENAI_API_KEY` + `LLM_MODEL`) e reinicie o app. Veja o `.env.example`."
        )
    else:
        left = max(SESSION_LIMIT - st.session_state.get("llm_calls", 0), 0)
        st.caption(f"LLM: `{client.provider}` / `{client.model}` · {left} chamadas restantes nesta sessão")
    return client


@st.cache_resource
def _daily_counter():
    from src.usage import DAILY_LIMIT, DailyCounter
    return DailyCounter(DAILY_LIMIT)


def llm_quota():
    """Consome uma chamada de LLM se ainda houver cota. Devolve (ok, mensagem)."""
    from src.usage import SESSION_LIMIT
    used = st.session_state.get("llm_calls", 0)
    if used >= SESSION_LIMIT:
        return False, (f"Esta sessão já usou as {SESSION_LIMIT} chamadas ao LLM permitidas na demo. "
                       "Rodando o projeto localmente não tem limite.")
    if not _daily_counter().try_consume():
        return False, "O limite diário de chamadas ao LLM da demo acabou. Amanhã ele volta."
    st.session_state.llm_calls = used + 1
    return True, ""


def metric(*path, default=None):
    """Lê um número de reports/metrics.json, para as legendas não ficarem desatualizadas."""
    from src.metrics_store import load_metrics
    node = load_metrics()
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def br(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


PROCESSED = DATA_PROCESSED
ART = ARTIFACTS
