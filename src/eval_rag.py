"""Avaliação das respostas do RAG.

Para cada pergunta de data/eval/rag_questions.json:

Checagens automáticas (não dependem de juiz):
  - % de frases/tópicos da resposta com pelo menos uma citação [R#]
  - citações inválidas (citou R9 tendo só 8 avaliações)
  - números na resposta que NÃO estão no bloco de números calculados
    (é a regra mais importante do prompt: o LLM não pode inventar número)

LLM como juiz:
  - faithfulness: cada afirmação é sustentada pelas avaliações citadas?
    (sim = 1, parcial = 0,5, não = 0)
  - relevância do contexto: das avaliações recuperadas, quantas tinham a ver
    com a pergunta
  - relevância da resposta: nota de 1 a 5

O juiz idealmente é um modelo diferente (ou maior) do que gerou a resposta.
Dá para trocar com JUDGE_MODEL no .env.

Rodar:  python -m src.eval_rag
"""
import json
import os
import re

from src.config import ROOT
from src.llm import get_client
from src.llm_extract import parse_json
from src.metrics_store import save_section
from src.rag import _format_numbers, _format_reviews, answer

QUESTIONS_FILE = ROOT / "data" / "eval" / "rag_questions.json"
OUT_FILE = ROOT / "artifacts" / "rag_eval_respostas.jsonl"

JUDGE_SYSTEM = """Você avalia respostas de um sistema de perguntas e respostas sobre avaliações de clientes.
Receberá a pergunta, as avaliações recuperadas (numeradas R1, R2...) e a resposta gerada.
Devolva SOMENTE um JSON:
{
  "claims": [{"text": "afirmação da resposta", "citations": [1, 2], "supported": "sim" | "parcial" | "nao"}],
  "relevant_reviews": [números das avaliações recuperadas que têm a ver com a pergunta],
  "answer_relevance": 1 a 5
}
Quebre a resposta em afirmações sobre o que os clientes relatam (ignore frases de transição).
"supported" diz se as avaliações CITADAS naquela afirmação sustentam o que ela diz. Sem citação = "nao".
Afirmações que só repetem os NÚMEROS fornecidos contam como "sim"."""

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _units(text):
    parts = [p.strip(" -*•\t") for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if len(p) > 25]


def _norm_number(n):
    """'3.917' -> '3917' (ponto de milhar) e '32,6' -> '32.6' (vírgula decimal).

    O LLM escreve os números do bloco no formato brasileiro; sem normalizar,
    "3.917" seria contado como número inventado.
    """
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", n):
        n = n.replace(".", "")
    return n.replace(",", ".")


def automatic_checks(res):
    units = _units(res["answer"])
    with_cite = [u for u in units if re.search(r"\[R\d+\]", u)]
    allowed = {_norm_number(n) for n in _NUMBER.findall(_format_numbers(res["numbers"]))}
    allowed |= {str(i) for i in range(1, len(res["hits"]) + 1)}  # os R1..R8
    allowed |= {"1", "2", "3", "4", "5"}  # "nota 1", "nota 2"...
    found = _NUMBER.findall(re.sub(r"\[R\d+\]", "", res["answer"]))
    unsupported = [n for n in found if _norm_number(n) not in allowed]
    return {
        "pct_unidades_com_citacao": round(len(with_cite) / len(units), 3) if units else 0.0,
        "citacoes_invalidas": len(res["invalid_citations"]),
        "numeros_nao_suportados": unsupported,
    }


def judge(res, client):
    prompt = (
        f"PERGUNTA: {res['question']}\n\n"
        f"AVALIAÇÕES RECUPERADAS:\n{_format_reviews(res['hits'])}\n\n"
        f"NÚMEROS FORNECIDOS AO SISTEMA:\n{_format_numbers(res['numbers'])}\n\n"
        f"RESPOSTA GERADA:\n{res['answer']}"
    )
    raw = client.complete(JUDGE_SYSTEM, prompt, max_tokens=8000, json_mode=True)
    try:
        obj = parse_json(raw)
    except ValueError:
        # juiz sem JSON válido: a pergunta fica sem nota em vez de interromper a
        # avaliação (as médias ignoram esses casos)
        return {"faithfulness": None, "n_afirmacoes": 0, "relevancia_contexto": None,
                "relevancia_resposta": None, "juiz_falhou": True}
    claims = obj.get("claims") or []
    pts = {"sim": 1.0, "parcial": 0.5, "nao": 0.0, "não": 0.0}
    faith = sum(pts.get(str(c.get("supported", "nao")).lower(), 0.0) for c in claims) / len(claims) if claims else 0.0
    relevant = [r for r in obj.get("relevant_reviews") or [] if isinstance(r, int)]
    return {
        "faithfulness": round(faith, 3),
        "n_afirmacoes": len(claims),
        "relevancia_contexto": round(len(set(relevant)) / max(len(res["hits"]), 1), 3),
        "relevancia_resposta": obj.get("answer_relevance"),
    }


def main():
    from src.index import ReviewIndex

    client = get_client()
    if client is None:
        raise SystemExit("A avaliação do RAG precisa de um LLM (ANTHROPIC_API_KEY ou OPENAI_API_KEY).")
    judge_client = get_client(model=os.getenv("JUDGE_MODEL")) if os.getenv("JUDGE_MODEL") else client

    index = ReviewIndex()
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    rows = []
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for q in questions:
            print(f"-> {q}")
            res = answer(q, index, client=client, k=8)
            row = {"pergunta": q, **automatic_checks(res), **judge(res, judge_client)}
            rows.append(row)
            f.write(json.dumps({**row, "resposta": res["answer"],
                                "avaliacoes": res["hits"]["review_text"].tolist()}, ensure_ascii=False) + "\n")
            print(f"   faithfulness={row['faithfulness']} contexto={row['relevancia_contexto']} "
                  f"citação={row['pct_unidades_com_citacao']} números_soltos={row['numeros_nao_suportados']}")

    def avg(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "gerador": client.model,
        "juiz": judge_client.model,
        "embeddings": index.backend,
        "n_perguntas": len(rows),
        "faithfulness_media": avg("faithfulness"),
        "relevancia_contexto_media": avg("relevancia_contexto"),
        "relevancia_resposta_media": avg("relevancia_resposta"),
        "pct_unidades_com_citacao_media": avg("pct_unidades_com_citacao"),
        "respostas_com_numero_inventado": sum(bool(r["numeros_nao_suportados"]) for r in rows),
        "citacoes_invalidas_total": sum(r["citacoes_invalidas"] for r in rows),
        "por_pergunta": rows,
    }
    save_section("rag", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "por_pergunta"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
