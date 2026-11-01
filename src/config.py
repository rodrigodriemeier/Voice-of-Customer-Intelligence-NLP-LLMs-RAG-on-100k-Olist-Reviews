from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ANNOTATIONS = ROOT / "data" / "annotations"
CACHE_DIR = ROOT / "data" / "cache"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

GOLD_FILE = ANNOTATIONS / "gold_400.csv"

RANDOM_STATE = 42

# Taxonomia das reclamações. As chaves em inglês são as que aparecem no JSON
# de saída; os rótulos em português são só para a interface.
CATEGORIES = {
    "delivery": "Entrega (atraso, não recebido, extravio)",
    "product_quality": "Qualidade / produto danificado",
    "wrong_product": "Produto errado ou diferente do anúncio",
    "missing_items": "Pedido incompleto / item faltando",
    "customer_service": "Atendimento e pós-venda (troca, reembolso, cancelamento)",
}
CATEGORY_KEYS = list(CATEGORIES)

SENTIMENTS = ["negative", "neutral", "positive"]
SEVERITIES = ["none", "low", "medium", "high"]

# Aspectos são "do que a pessoa está falando", independente de ser elogio ou crítica.
ASPECTS = ["entrega", "produto", "embalagem", "atendimento", "preco", "loja"]

# Lido do ambiente para dar para trocar sem mexer no código.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sbert")  # sbert | lsa
SBERT_MODEL = os.getenv(
    "SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def score_to_sentiment(score: int) -> str:
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"
