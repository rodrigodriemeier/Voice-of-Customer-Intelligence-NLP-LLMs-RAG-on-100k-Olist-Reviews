from src.text import build_review_text, normalize


def test_normalize_tira_acento_e_pontuacao():
    assert normalize("Não CHEGOU!!!") == "nao chegou"


def test_normalize_anonimiza_lojas():
    assert normalize("comprei nas Lojas Lannister") == "comprei nas lojas loja"
    assert "stark" not in normalize("o stark esta de parabens")


def test_normalize_aguenta_nan():
    assert normalize(None) == ""
    assert normalize(float("nan")) == ""


def test_titulo_repetido_nao_duplica():
    assert build_review_text("Recomendo", "Recomendo, chegou rápido") == "Recomendo, chegou rápido"
    assert build_review_text("Ótimo", "chegou rápido") == "Ótimo. chegou rápido"
    assert build_review_text(None, "só a mensagem") == "só a mensagem"
