from datetime import date

import pandas as pd

from src.eval_gold import agreement
from src.report import gold_review_block
from src.usage import DailyCounter


def test_contador_diario_bloqueia_e_zera_no_dia_seguinte():
    hoje = [date(2026, 11, 1)]
    c = DailyCounter(2, today=lambda: hoje[0])
    assert c.try_consume() and c.try_consume()
    assert not c.try_consume()
    hoje[0] = date(2026, 11, 2)
    assert c.try_consume()
    assert c.remaining == 1


def test_concordancia():
    rev = pd.DataFrame({
        "review_id": ["a", "b", "c", "d"],
        "sentiment": ["negative", "positive", "neutral", "negative"],
        "severity": ["high", "none", "low", "medium"],
        "categories": ["delivery", "", "delivery", "product_quality|customer_service"],
    })
    first = pd.DataFrame({
        "review_id": ["a", "b", "c", "d"],
        "sentiment": ["negative", "positive", "negative", "negative"],
        "severity": ["high", "none", "low", "high"],
        "categories": ["delivery", "", "delivery", "product_quality"],
    })
    a = agreement(rev, first)
    assert a["n"] == 4
    assert a["sentimento"]["concordancia"] == 0.75
    assert a["linhas_com_alguma_mudanca"] == 2
    assert a["categorias"]["conjunto_identico"] == 0.75
    assert a["categorias"]["por_categoria"]["delivery"]["concordancia"] == 1.0


def test_bloco_do_readme_sem_revisao_nao_finge_que_revisou():
    txt = gold_review_block({"revisao_gold": {"n_total": 400, "n_revisadas": 0, "n_revisadas_cegas": 0}})
    assert "ainda não feita" in txt
    txt = gold_review_block({})
    assert "ainda não feita" in txt
