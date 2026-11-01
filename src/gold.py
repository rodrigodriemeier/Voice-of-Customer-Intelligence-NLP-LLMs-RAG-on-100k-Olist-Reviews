import pandas as pd

from src.config import GOLD_FILE


def split_list(value):
    if not isinstance(value, str) or not value.strip():
        return []
    return [v for v in value.split("|") if v]


def load_gold(only_reviewed=False):
    gold = pd.read_csv(GOLD_FILE, keep_default_na=False, dtype={"review_score": int})
    gold["reviewed"] = gold["reviewed"].astype(str).str.lower().isin(["true", "1", "sim"])
    # "cega" ou "com_rotulos"; a coluna só aparece depois da primeira revisão
    if "review_mode" not in gold.columns:
        gold["review_mode"] = ""
    gold["categories_list"] = gold["categories"].map(split_list)
    gold["aspects_list"] = gold["aspects"].map(split_list)
    if only_reviewed:
        gold = gold[gold["reviewed"]]
    return gold.reset_index(drop=True)
