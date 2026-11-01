import pandas as pd

from src.rag import Filters, cited_ids, filters_from_question, search_query


def test_filtros_da_pergunta():
    f = filters_from_question("Quais são os principais problemas de entrega de produtos eletrônicos?")
    assert f.macro_categories == ["Eletrônicos e informática"]
    assert f.only_problems


def test_pergunta_neutra_nao_filtra_nota():
    f = filters_from_question("O que os clientes elogiam na entrega?")
    assert not f.only_problems


def test_busca_tira_a_moldura_da_pergunta():
    q = "Quais são os principais problemas relacionados à entrega de produtos eletrônicos?"
    assert search_query(q, filters_from_question(q)) == "entrega"


def test_citacoes():
    valid, invalid = cited_ids("Atrasos [R1][R3] e um erro [R9]", n_hits=8)
    assert valid == [1, 3]
    assert invalid == [9]


def test_mask():
    meta = pd.DataFrame({
        "macro_category": ["A", "B", "A"],
        "review_score": [1, 2, 5],
        "review_creation_date": pd.to_datetime(["2018-01-01"] * 3),
        "customer_state": ["SP", "RJ", "SP"],
    })
    assert Filters(macro_categories=["A"], only_problems=True).mask(meta).tolist() == [True, False, False]
