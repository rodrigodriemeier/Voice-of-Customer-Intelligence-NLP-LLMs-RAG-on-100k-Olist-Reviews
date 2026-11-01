"""Cliente mínimo de LLM, com cache em disco.

Suporta dois "provedores":
  - anthropic: API da Anthropic (Claude)
  - openai:    qualquer API compatível com OpenAI (OpenAI, Groq, OpenRouter,
               Ollama local em http://localhost:11434/v1 ...)

Configuração por variável de ambiente (ver .env.example).

O cache evita pagar de novo por chamadas idênticas ao reexecutar as avaliações.
"""
import hashlib
import json
import os
import time

from src.config import CACHE_DIR

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv é opcional
    pass

# Sem modelo padrão para "openai": depende do serviço (OpenAI, Groq, Ollama...).
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": None,
}


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider=None, model=None, use_cache=True):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or self._guess_provider()).lower()
        if self.provider not in DEFAULT_MODELS:
            raise LLMError(f"LLM_PROVIDER inválido: {self.provider!r} (use anthropic ou openai)")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        if not self.model:
            raise LLMError("Defina LLM_MODEL (ex.: llama3.1:8b no Ollama).")
        self.use_cache = use_cache
        self._cache_file = CACHE_DIR / "llm_cache.jsonl"
        self._cache = self._load_cache() if use_cache else {}
        self._client = self._make_client()

    @staticmethod
    def _guess_provider():
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
            return "openai"
        raise LLMError("Nenhuma chave de LLM configurada (ANTHROPIC_API_KEY ou OPENAI_API_KEY).")

    def _make_client(self):
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic()
        import openai
        # Ollama não precisa de chave, mas o SDK exige alguma string
        return openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )

    # --- cache -------------------------------------------------------------

    def _load_cache(self):
        cache = {}
        if self._cache_file.exists():
            for line in self._cache_file.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    cache[item["key"]] = item["response"]
                except (json.JSONDecodeError, KeyError):
                    continue  # linha corrompida (processo morto no meio da escrita)
        return cache

    def _key(self, system, user, max_tokens):
        raw = json.dumps([self.provider, self.model, system, user, max_tokens], ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _store(self, key, response):
        self._cache[key] = response
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")

    # --- chamada -------------------------------------------------------------

    def complete(self, system, user, max_tokens=800, json_mode=False):
        key = self._key(system, user, max_tokens)
        if self.use_cache and key in self._cache:
            return self._cache[key]

        last_err = None
        for attempt in range(4):
            try:
                text, truncated = self._call(system, user, max_tokens, json_mode)
                # Modelos com raciocínio ativo (padrão no Haiku 5.5) gastam parte do
                # max_tokens raciocinando, e a resposta pode vir truncada. Nesse
                # caso a chamada é repetida com limite maior.
                budget = max_tokens
                while truncated and budget < 16_000:
                    budget = min(budget * 4, 16_000)
                    print(f"   [llm] resposta cortada, refazendo com max_tokens={budget}")
                    text, truncated = self._call(system, user, budget, json_mode)
                if self.use_cache:
                    self._store(key, text)
                return text
            except Exception as err:  # rate limit, timeout, 5xx...
                # chave errada / modelo que não existe: tentar de novo não resolve
                if any(n in type(err).__name__ for n in ("Authentication", "PermissionDenied", "NotFound", "BadRequest")):
                    raise LLMError(f"{type(err).__name__}: {err}") from err
                last_err = err
                wait = 2 ** attempt
                print(f"   [llm] erro ({type(err).__name__}), tentando de novo em {wait}s")
                time.sleep(wait)
        raise LLMError(f"Falhou depois de 4 tentativas: {last_err}") from last_err

    def _call(self, system, user, max_tokens, json_mode):
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            # só blocos de texto: a resposta pode começar com blocos de "thinking"
            text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
            return text, resp.stop_reason == "max_tokens"

        import openai

        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as err:
            # modelos de raciocínio da OpenAI recusam max_tokens e temperature=0
            # (Ollama/Groq aceitam); repete com max_completion_tokens
            if "max_tokens" not in str(err) and "temperature" not in str(err):
                raise
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            kwargs.pop("temperature")
            resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return choice.message.content or "", choice.finish_reason == "length"

    def __repr__(self):
        return f"LLMClient({self.provider}:{self.model})"


def get_client(**kw):
    """Devolve um cliente ou None se não tiver nada configurado.

    O app usa isso para decidir se mostra as partes que dependem de LLM.
    """
    try:
        return LLMClient(**kw)
    except LLMError:
        return None
