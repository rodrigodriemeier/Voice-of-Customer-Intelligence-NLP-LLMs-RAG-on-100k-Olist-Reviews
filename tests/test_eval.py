from src.eval_llm import token_f1


def test_token_f1():
    assert token_f1("veio quebrado", "veio quebrado") == 1.0
    assert token_f1("", "x") == 0.0
    assert 0 < token_f1("o produto veio quebrado", "veio quebrado na caixa") < 1
    # acento não pode atrapalhar
    assert token_f1("nao chegou", "não chegou") == 1.0


def test_resposta_sem_json_conta_como_erro_e_nao_quebra():
    from src.eval_llm import evaluate_llm
    from src.gold import load_gold
    gold = load_gold().head(3)
    preds = [{"review_id": r, "error": "json inválido", "raw": "..."} for r in gold["review_id"]]
    m = evaluate_llm(preds, model_name="teste", save=False)
    assert m["respostas_sem_json_valido"] == 3


def test_numero_com_ponto_de_milhar_nao_conta_como_inventado():
    import pandas as pd
    from src.eval_rag import automatic_checks
    res = {
        "answer": "Há 3.917 avaliações com problema e 32,6% são de entrega [R1]. Outro dado: 77% [R2].",
        "numbers": {"avaliacoes_com_texto_no_recorte": 10643, "nota_media": 3.64, "pct_nota_1_ou_2": 26.5,
                    "pct_entregues_com_atraso": 9.4, "avaliacoes_com_problema": 3917,
                    "pct_categoria_entre_problemas": {"Entrega": 32.6}},
        "hits": pd.DataFrame({"x": range(8)}),
        "invalid_citations": [],
    }
    assert automatic_checks(res)["numeros_nao_suportados"] == ["77"]
