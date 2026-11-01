"""Etapa 2B: extração estruturada com LLM.

Para cada avaliação o LLM devolve um JSON com sentimento, categorias,
severidade, aspectos (com evidência) e uma descrição curta do problema.
A saída é validada aqui, porque o modelo pode desviar do formato pedido.

Rodar no gold (gera artifacts/gold_pred_llm.jsonl e as métricas):
    python -m src.llm_extract --gold
"""
import argparse
import json
import re
import time

from src.config import ARTIFACTS, ASPECTS, CATEGORIES, CATEGORY_KEYS, SENTIMENTS, SEVERITIES
from src.text import clean_display

PRED_FILE = ARTIFACTS / "gold_pred_llm.jsonl"

_cat_lines = "\n".join(f'  - "{k}": {v}' for k, v in CATEGORIES.items())

SYSTEM_PROMPT = f"""Você analisa avaliações de clientes de um marketplace brasileiro (Olist).
Leia a avaliação e devolva SOMENTE um objeto JSON, sem texto antes ou depois, neste formato:

{{
  "sentiment": "negative" | "neutral" | "positive",
  "categories": [lista de categorias de reclamação, pode ser vazia],
  "severity": "none" | "low" | "medium" | "high",
  "aspects": [{{"aspect": "...", "sentiment": "negative" | "neutral" | "positive", "evidence": "..."}}],
  "problem": "frase curta descrevendo o problema" ou null
}}

Categorias possíveis (use só estas chaves):
{_cat_lines}

Regras:
- sentiment é o tom geral do TEXTO. "Ainda não recebi" é negative. Elogio com ressalva pequena é positive. Misto equilibrado ou só factual é neutral.
- Marque uma categoria sempre que o texto relatar aquele problema, mesmo numa avaliação positiva. Reclamação só de preço ou frete caro não tem categoria.
- severity: none = sem problema; low = incômodo pequeno ou resolvido; medium = problema real mas o cliente tem o produto e consegue usar; high = não recebeu, veio quebrado/inutilizável, produto totalmente errado, item pago que não veio, cobrança indevida, menção a Procon/golpe.
- aspect deve ser um destes: {", ".join(ASPECTS)}. "loja" é a loja/site/vendedor em geral.
- evidence tem que ser um trecho COPIADO LITERALMENTE da avaliação (mesmas palavras, mesma grafia, inclusive erros de digitação). Não resuma, não corrija.
- problem: null quando não há problema."""

FEW_SHOT = [
    (
        "Meu pedido chegou três dias atrasado e o produto veio danificado.",
        {
            "sentiment": "negative",
            "categories": ["delivery", "product_quality"],
            "severity": "high",
            "aspects": [
                {"aspect": "entrega", "sentiment": "negative", "evidence": "chegou três dias atrasado"},
                {"aspect": "produto", "sentiment": "negative", "evidence": "o produto veio danificado"},
            ],
            "problem": "Entrega atrasada e produto danificado",
        },
    ),
    (
        "chegou antes do prazo, bem embalado. so achei a cor um pouco mais escura q na foto",
        {
            "sentiment": "positive",
            "categories": ["wrong_product"],
            "severity": "low",
            "aspects": [
                {"aspect": "entrega", "sentiment": "positive", "evidence": "chegou antes do prazo"},
                {"aspect": "embalagem", "sentiment": "positive", "evidence": "bem embalado"},
                {"aspect": "produto", "sentiment": "negative", "evidence": "achei a cor um pouco mais escura q na foto"},
            ],
            "problem": "Cor um pouco diferente da foto",
        },
    ),
]


def build_user_prompt(review_text):
    parts = []
    for text, answer in FEW_SHOT:
        parts.append(f"Avaliação: {text}\nJSON: {json.dumps(answer, ensure_ascii=False)}")
    parts.append(f"Avaliação: {clean_display(review_text)}\nJSON:")
    return "\n\n".join(parts)


# --- parsing e validação -------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(raw):
    """Acha o primeiro objeto JSON na resposta. Levanta ValueError se não achar."""
    text = _FENCE.sub("", raw.strip())
    start = text.find("{")
    if start == -1:
        raise ValueError("resposta sem JSON")
    # raw_decode para no fim do primeiro objeto e ignora lixo depois dele
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("JSON não é um objeto")
    return obj


def _squash(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def is_grounded(evidence, review_text):
    """A evidência aparece mesmo no texto? (ignora caixa e espaços)"""
    if not evidence:
        return False
    return _squash(evidence) in _squash(review_text)


def validate(obj, review_text):
    """Normaliza o JSON do LLM. Valores fora do esperado viram o padrão e
    ficam registrados em `issues`, para medir quanto o modelo desvia do formato."""
    issues = []

    sentiment = obj.get("sentiment")
    if sentiment not in SENTIMENTS:
        issues.append(f"sentiment inválido: {sentiment!r}")
        sentiment = "neutral"

    cats = obj.get("categories") or []
    if isinstance(cats, str):
        cats = [cats]
    clean_cats = []
    for c in cats:
        if c in CATEGORY_KEYS and c not in clean_cats:
            clean_cats.append(c)
        else:
            issues.append(f"categoria inválida: {c!r}")

    severity = obj.get("severity")
    if severity not in SEVERITIES:
        issues.append(f"severity inválida: {severity!r}")
        severity = "medium" if clean_cats else "none"

    aspects = []
    for a in obj.get("aspects") or []:
        if not isinstance(a, dict):
            continue
        name = str(a.get("aspect", "")).lower().strip()
        if name == "preço":
            name = "preco"
        if name not in ASPECTS:
            issues.append(f"aspecto inválido: {name!r}")
            continue
        ev = str(a.get("evidence") or "")
        aspects.append({
            "aspect": name,
            "sentiment": a.get("sentiment") if a.get("sentiment") in SENTIMENTS else "neutral",
            "evidence": ev,
            "grounded": is_grounded(ev, review_text),
        })

    problem = obj.get("problem")
    if isinstance(problem, str) and problem.strip().lower() in ("", "null", "none", "nenhum"):
        problem = None

    return {
        "sentiment": sentiment,
        "categories": clean_cats,
        "severity": severity,
        "aspects": aspects,
        "problem": problem,
        "issues": issues,
    }


def extract(review_text, client):
    user = build_user_prompt(review_text)
    raw = client.complete(SYSTEM_PROMPT, user, max_tokens=600, json_mode=True)
    try:
        obj = parse_json(raw)
    except ValueError:
        # Em geral é resposta truncada: o raciocínio do modelo consome o
        # max_tokens antes de o JSON terminar. Repete com limite maior.
        raw = client.complete(SYSTEM_PROMPT, user, max_tokens=4000, json_mode=True)
        try:
            obj = parse_json(raw)
        except ValueError as err:
            return {"error": f"json inválido: {err}", "raw": raw}
    out = validate(obj, review_text)
    out["raw"] = raw
    return out


def run_on_gold(client, limit=None):
    from src.gold import load_gold

    gold = load_gold()
    if limit:
        gold = gold.head(limit)

    done = {}
    if PRED_FILE.exists():
        for line in PRED_FILE.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            # respostas com erro não contam como feitas: são refeitas
            if item.get("model") == client.model and "error" not in item:
                done[item["review_id"]] = item

    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with PRED_FILE.open("a", encoding="utf-8") as f:
        for i, row in gold.iterrows():
            if row["review_id"] in done:
                continue
            res = extract(row["review_text"], client)
            res.update(review_id=row["review_id"], model=client.model)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            done[row["review_id"]] = res
            if (i + 1) % 25 == 0:
                print(f"   {i + 1}/{len(gold)} ({time.time() - t0:.0f}s)")
    others = []
    if PRED_FILE.exists():
        others = [line for line in PRED_FILE.read_text(encoding="utf-8").splitlines()
                  if json.loads(line).get("model") != client.model]
    PRED_FILE.write_text("\n".join(others + [json.dumps(x, ensure_ascii=False) for x in done.values()]) + "\n",
                         encoding="utf-8")
    return [done[r] for r in gold["review_id"] if r in done]


def main():
    from src.eval_llm import evaluate_llm
    from src.llm import get_client

    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true", help="roda nas 400 avaliações do gold e avalia")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--text", type=str, default=None, help="testa uma avaliação avulsa")
    args = ap.parse_args()

    client = get_client()
    if client is None:
        raise SystemExit("Configure ANTHROPIC_API_KEY ou OPENAI_API_KEY (veja .env.example).")
    print(f"Usando {client}")

    if args.text:
        res = extract(args.text, client)
        res.pop("raw", None)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    if args.gold:
        preds = run_on_gold(client, limit=args.limit)
        evaluate_llm(preds, model_name=client.model, save=args.limit is None)


if __name__ == "__main__":
    main()
