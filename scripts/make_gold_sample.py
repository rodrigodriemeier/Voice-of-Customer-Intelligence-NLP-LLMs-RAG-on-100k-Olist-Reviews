"""Sorteia as 400 avaliações do conjunto de referência (gold).

A amostra é estratificada pela nota e tem mais avaliações ruins do que a
distribuição real (onde ~58% são nota 5). O objetivo é medir se o sistema
acerta o *tipo* de reclamação; numa amostra aleatória simples, as categorias
menores (produto errado, item faltando) teriam cerca de 20 exemplos cada.

Consequência: as métricas no gold NÃO representam a distribuição real. Para
isso serve o holdout aleatório do classificador.

Rodar só uma vez:  python scripts/make_gold_sample.py
(não sobrescreve se o arquivo já existir, para não perder a anotação)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_PROCESSED, GOLD_FILE, RANDOM_STATE  # noqa: E402

PER_SCORE = {1: 120, 2: 50, 3: 70, 4: 60, 5: 100}

LABEL_COLS = ["sentiment", "categories", "severity", "aspects", "evidence", "problem", "notes"]


def main():
    if GOLD_FILE.exists() and "--force" not in sys.argv:
        print(f"{GOLD_FILE} já existe, não vou sobrescrever (use --force).")
        return

    corpus = pd.read_parquet(DATA_PROCESSED / "corpus.parquet")
    # textos de 1-3 caracteres ("." / "ok") não têm conteúdo anotável
    corpus = corpus[corpus["review_text"].str.len() >= 4]

    parts = [
        corpus[corpus["review_score"] == score].sample(n, random_state=RANDOM_STATE)
        for score, n in PER_SCORE.items()
    ]
    gold = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)
    gold = gold[["review_id", "review_score", "macro_category", "review_text"]].reset_index(drop=True)
    for col in LABEL_COLS:
        gold[col] = ""
    gold["reviewed"] = False

    GOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(GOLD_FILE, index=False)
    print(f"{len(gold)} avaliações salvas em {GOLD_FILE}")


if __name__ == "__main__":
    main()
