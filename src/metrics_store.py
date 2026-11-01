"""Guarda as métricas de cada etapa num JSON só (reports/metrics.json).

Cada script escreve a sua seção; assim dá para rodar as etapas em qualquer
ordem e o report final junta tudo.
"""
import json
from datetime import datetime

from src.config import REPORTS

METRICS_FILE = REPORTS / "metrics.json"


def load_metrics():
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return {}


def save_section(name, payload):
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    payload = dict(payload)
    payload["_atualizado_em"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    data[name] = payload
    METRICS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
