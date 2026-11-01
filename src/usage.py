"""Limite de chamadas de LLM para o app público.

Com o app público, cada clique em "Perguntar" consome a chave de API do
projeto. Dois limites, configuráveis por variável de ambiente (ou nos
Secrets do Streamlit Cloud):

  LLM_SESSION_LIMIT  chamadas por sessão do navegador (padrão 10)
  LLM_DAILY_LIMIT    chamadas por dia somando todo mundo (padrão 200)

O contador diário vive na memória do processo, então zera se o app
reiniciar. Para uma demo isso basta; o controle definitivo é o limite de
gasto configurado no painel do provedor.
"""
import os
import threading
from datetime import date


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


SESSION_LIMIT = _env_int("LLM_SESSION_LIMIT", 10)
DAILY_LIMIT = _env_int("LLM_DAILY_LIMIT", 200)


class DailyCounter:
    def __init__(self, limit, today=date.today):
        self.limit = limit
        self._today = today
        self._day = today()
        self._count = 0
        self._lock = threading.Lock()

    def try_consume(self):
        with self._lock:
            if self._today() != self._day:
                self._day, self._count = self._today(), 0
            if self._count >= self.limit:
                return False
            self._count += 1
            return True

    @property
    def remaining(self):
        with self._lock:
            return max(self.limit - self._count, 0) if self._today() == self._day else self.limit
