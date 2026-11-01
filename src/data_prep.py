"""Etapa 1: junta avaliações, pedidos, itens, produtos e clientes numa tabela só.

Saídas (em data/processed/):
    reviews.parquet  -> uma linha por par (review_id, order_id), com as features do pedido
    corpus.parquet   -> só avaliações com texto, uma linha por review_id (base do NLP e do RAG)

Rodar:  python -m src.data_prep
"""
import pandas as pd

from src.categories import pretty, to_macro
from src.config import DATA_PROCESSED, DATA_RAW, score_to_sentiment
from src.text import build_review_text, normalize

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def _read(name, **kw):
    path = DATA_RAW / f"olist_{name}_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Não achei {path}. Rode antes: python scripts/download_data.py"
        )
    return pd.read_csv(path, **kw)


def load_orders():
    orders = _read("orders", parse_dates=DATE_COLS)
    delivered = orders["order_delivered_customer_date"]
    estimated = orders["order_estimated_delivery_date"]

    orders["delivery_days"] = (delivered - orders["order_purchase_timestamp"]).dt.total_seconds() / 86400
    # a data estimada vem sempre à meia-noite: a comparação é por dia, senão um
    # pedido entregue às 15h do dia previsto contaria como atrasado
    orders["delay_days"] = (delivered.dt.normalize() - estimated).dt.days
    orders["was_delivered"] = delivered.notna()
    # boolean "nullable": pedido não entregue não é nem atrasado nem no prazo
    orders["is_late"] = (orders["delay_days"] > 0).astype("boolean")
    orders.loc[~orders["was_delivered"], "is_late"] = pd.NA
    orders["purchase_month"] = orders["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    return orders


def load_items():
    items = _read("order_items")
    products = _read("products", usecols=["product_id", "product_category_name"])
    items = items.merge(products, on="product_id", how="left")

    # Em pedidos com vários itens, a categoria do pedido é a do item mais caro.
    # (A moda não serve: com dois itens de categorias diferentes, o desempate
    # seria por ordem alfabética.)
    main = (
        items.sort_values(["order_id", "price"], ascending=[True, False])
        .drop_duplicates("order_id")[["order_id", "product_category_name"]]
    )
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        n_categories=("product_category_name", "nunique"),
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
    )
    out = agg.reset_index().merge(main, on="order_id", how="left")
    out["product_category"] = out["product_category_name"].map(pretty)
    out["macro_category"] = out["product_category_name"].map(to_macro)
    out["freight_ratio"] = out["freight_value"] / out["order_value"]
    return out


def build_reviews():
    reviews = _read("order_reviews", parse_dates=["review_creation_date", "review_answer_timestamp"])
    customers = _read("customers", dtype={"customer_zip_code_prefix": str})
    customers["customer_zip_code_prefix"] = customers["customer_zip_code_prefix"].str.zfill(5)

    orders = load_orders()
    items = load_items()

    df = (
        reviews.merge(orders, on="order_id", how="left")
        .merge(items, on="order_id", how="left")
        .merge(customers[["customer_id", "customer_state", "customer_city"]], on="customer_id", how="left")
    )

    df["review_text"] = [
        build_review_text(t, m) for t, m in zip(df["review_comment_title"], df["review_comment_message"])
    ]
    df["has_text"] = df["review_text"].str.len() > 0
    df["text_norm"] = df["review_text"].map(normalize)
    df["sentiment_from_score"] = df["review_score"].map(score_to_sentiment)
    df["is_negative"] = df["review_score"] <= 2

    df["delay_bucket"] = pd.cut(
        df["delay_days"],
        bins=[-1000, -7, 0, 3, 7, 14, 1000],
        labels=["7+ dias adiantado", "0-6 dias adiantado", "1-3 dias atrasado",
                "4-7 dias atrasado", "8-14 dias atrasado", "15+ dias atrasado"],
    ).astype("string")
    df.loc[~df["was_delivered"].fillna(False).astype(bool), "delay_bucket"] = "não entregue"
    return df


def build_corpus(reviews):
    """Avaliações com texto, sem repetir review_id.

    Existem ~800 review_id ligados a mais de um pedido (a mesma avaliação
    copiada para pedidos feitos juntos). Para texto isso só duplicaria o dado.
    """
    corpus = (
        reviews[reviews["has_text"]]
        .sort_values("review_creation_date")
        .drop_duplicates("review_id")
        .reset_index(drop=True)
    )
    cols = [
        "review_id", "order_id", "review_score", "review_text", "text_norm",
        "sentiment_from_score", "review_creation_date", "order_status",
        "delay_days", "is_late", "was_delivered", "delay_bucket",
        "product_category_name", "product_category", "macro_category",
        "n_items", "n_sellers", "customer_state", "purchase_month",
    ]
    return corpus[cols]


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    reviews = build_reviews()
    corpus = build_corpus(reviews)

    # o texto original fica só no corpus, para não duplicar dado
    slim = reviews.drop(columns=["text_norm", "review_comment_title", "review_comment_message", "customer_city"])
    slim.to_parquet(DATA_PROCESSED / "reviews.parquet", index=False)
    corpus.to_parquet(DATA_PROCESSED / "corpus.parquet", index=False)

    print(f"reviews.parquet: {len(reviews):,} linhas")
    print(f"corpus.parquet:  {len(corpus):,} avaliações com texto")
    late = reviews["is_late"].astype("boolean")
    print(f"pedidos entregues com atraso: {late.mean():.1%}")


if __name__ == "__main__":
    main()
