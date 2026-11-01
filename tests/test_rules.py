import pytest

from src.rules import label_categories, severity_from_rules


@pytest.mark.parametrize("texto, esperado", [
    ("Não recebi o produto", {"delivery"}),
    ("chegou antes do prazo, adorei", set()),
    ("não houve atraso nenhum", set()),
    ("Comprei duas unidades e só veio uma", {"missing_items"}),
    ("veio quebrado e ninguém responde", {"product_quality", "customer_service"}),
    ("recebi o produto errado", {"wrong_product"}),
])
def test_categorias(texto, esperado):
    assert set(label_categories(texto)) == esperado


def test_falta_de_contato_nao_e_item_faltando():
    assert "missing_items" not in label_categories("falta de contato da loja")


def test_severidade():
    assert severity_from_rules("não recebi até hoje", ["delivery"], "negative") == "high"
    assert severity_from_rules("demorou um pouco mas chegou", ["delivery"], "positive") == "low"
    assert severity_from_rules("adorei", [], "positive") == "none"
    assert severity_from_rules("chegou atrasado e danificado", ["delivery", "product_quality"], "negative") == "high"
