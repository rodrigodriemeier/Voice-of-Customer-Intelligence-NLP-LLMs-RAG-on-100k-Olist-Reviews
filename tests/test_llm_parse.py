import pytest

from src.llm_extract import is_grounded, parse_json, validate

TEXTO = "Meu pedido chegou três dias atrasado e o produto veio danificado."


def test_parse_json_com_cerca_e_lixo():
    raw = 'Claro! ```json\n{"sentiment": "negative"}\n``` espero ter ajudado {"outro": 1}'
    assert parse_json(raw) == {"sentiment": "negative"}


def test_parse_json_sem_json():
    with pytest.raises(ValueError):
        parse_json("desculpe, não consigo")


def test_validate_corrige_valores_fora_do_esperado():
    out = validate({
        "sentiment": "muito negativo",
        "categories": ["delivery", "frete", "delivery"],
        "severity": "altíssima",
        "aspects": [{"aspect": "Preço", "sentiment": "negative", "evidence": "caro"},
                    {"aspect": "cor", "evidence": "x"}],
        "problem": "null",
    }, TEXTO)
    assert out["sentiment"] == "neutral"
    assert out["categories"] == ["delivery"]
    assert out["severity"] == "medium"
    assert [a["aspect"] for a in out["aspects"]] == ["preco"]
    assert out["problem"] is None
    assert len(out["issues"]) >= 4


def test_evidencia_literal():
    assert is_grounded("chegou  três dias ATRASADO", TEXTO)
    assert not is_grounded("chegou com 3 dias de atraso", TEXTO)
