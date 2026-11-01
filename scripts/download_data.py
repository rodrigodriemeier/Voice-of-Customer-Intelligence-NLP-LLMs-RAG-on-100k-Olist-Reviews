"""Baixa os CSVs do dataset da Olist para data/raw/.

O dataset do Kaggle (olistbr/brazilian-ecommerce) também é publicado pela
Olist no GitHub (github.com/olist/work-at-olist-data), que não exige login
nem token. O conteúdo é o mesmo; a única diferença é que no Kaggle o CEP vem
entre aspas, com zero à esquerda (o data_prep.py normaliza isso).

Alternativa: baixar o zip do Kaggle e descompactar em data/raw/.

Rodar:  python scripts/download_data.py
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"
FILES = [
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
    # geolocation (60 MB) e payments não são usados no projeto
]

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW / name
        if dest.exists() and "--force" not in sys.argv:
            print(f"ok  {name} (já existe)")
            continue
        print(f"... {name}", end="", flush=True)
        urllib.request.urlretrieve(BASE + name, dest)
        print(f"\r ok {name} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
