"""Etapa 3b: perguntas em linguagem natural respondidas com avaliações reais.

Fluxo:
  1. lê a pergunta e tira filtros óbvios (macrocategoria, "problemas" ->
     só avaliações com nota <= 3)
  2. busca semântica no FAISS dentro do subconjunto filtrado
  3. calcula com pandas alguns números do subconjunto (o LLM NÃO inventa
     número; ele só cita os que recebe)
  4. o LLM escreve a resposta citando [R1], [R2]... das avaliações recuperadas

Sem LLM configurado, o passo 4 vira um resumo simples feito em Python, para
o app continuar útil.
"""
import re
from dataclasses import dataclass, field

import pandas as pd

from src.categories import MACRO_KEYWORDS
from src.config import CATEGORIES
from src.text import strip_accents

_PROBLEM_WORDS = re.compile(
    r"\b(problema|reclama|insatisf|ruim|pior|negativ|defeit|atras|demor|falh|queixa|critica|nao recebi|errad|quebr)\w*"
)

# Palavras de "moldura" da pergunta, removidas antes da busca. Sem isso,
# termos genéricos como "produtos" dominam a similaridade. Em "Quais são os
# principais problemas relacionados à entrega de produtos eletrônicos?",
# sobra "entrega", que é o assunto de fato.
_QUESTION_FRAME = {
    "quais", "qual", "sao", "e", "os", "as", "o", "a", "principais", "maiores", "mais",
    "comuns", "frequentes", "problemas", "problema", "reclamacoes", "reclamacao",
    "queixas", "relacionados", "relacionadas", "relacionado", "relacionada", "com",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "sobre", "clientes",
    "cliente", "consumidores", "produtos", "produto", "categoria", "que", "como", "por",
    "motivos", "causas", "ha", "existem", "tem", "dizem", "falam", "relatam", "aos",
    "ao", "para", "pra", "quando", "onde", "itens", "item",
}


def search_query(question, filters):
    q = strip_accents(question.lower())
    words = re.findall(r"\w+", q)
    macro_words = [w for m in filters.macro_categories for w in MACRO_KEYWORDS.get(m, [])]
    kept = [w for w in words if w not in _QUESTION_FRAME and not any(w.startswith(mw.strip()) for mw in macro_words)]
    return " ".join(kept) if kept else question


RAG_SYSTEM = """Você é um analista de experiência do cliente. Responda à pergunta usando SOMENTE as avaliações fornecidas.

Regras:
- Toda afirmação sobre o que os clientes relatam precisa citar as avaliações que a sustentam, no formato [R1], [R2].
- Não invente números nem porcentagens. Se quiser falar em quantidade, use apenas os números do bloco NÚMEROS, que foram calculados sobre a base inteira, e deixe claro que vêm de lá. Não faça contas com as avaliações recuperadas, elas são só uma amostra.
- Se as avaliações não forem suficientes para responder, diga isso.
- Escreva em português, de forma direta: um parágrafo curto de visão geral e depois os principais problemas em tópicos (no máximo 5)."""


@dataclass
class Filters:
    macro_categories: list = field(default_factory=list)
    only_problems: bool = False
    date_from: object = None
    date_to: object = None
    states: list = field(default_factory=list)

    def mask(self, meta):
        m = pd.Series(True, index=meta.index)
        if self.macro_categories:
            m &= meta["macro_category"].isin(self.macro_categories)
        if self.only_problems:
            m &= meta["review_score"] <= 3
        if self.date_from is not None:
            m &= meta["review_creation_date"] >= pd.Timestamp(self.date_from)
        if self.date_to is not None:
            m &= meta["review_creation_date"] <= pd.Timestamp(self.date_to)
        if self.states:
            m &= meta["customer_state"].isin(self.states)
        return m.to_numpy()

    def describe(self):
        parts = []
        if self.macro_categories:
            parts.append("categorias: " + ", ".join(self.macro_categories))
        if self.only_problems:
            parts.append("só avaliações com nota <= 3")
        if self.states:
            parts.append("estados: " + ", ".join(self.states))
        if self.date_from is not None or self.date_to is not None:
            parts.append(f"período: {self.date_from or '...'} a {self.date_to or '...'}")
        return "; ".join(parts) or "sem filtros"


def filters_from_question(question):
    """Heurística simples. Os filtros aparecem preenchidos na interface e
    podem ser ajustados."""
    q = " " + strip_accents(question.lower()) + " "
    macros = [m for m, words in MACRO_KEYWORDS.items() if any(w in q for w in words)]
    return Filters(macro_categories=macros, only_problems=bool(_PROBLEM_WORDS.search(q)))


def subset_numbers(meta, filters):
    """Números calculados aqui, com pandas (não pelo LLM).

    As taxas usam o recorte SEM o filtro de "só problemas"; com ele, a
    porcentagem de nota baixa seria inflada pelo próprio filtro. A
    distribuição das categorias usa só as avaliações com problema
    (nota <= 3) do recorte.
    """
    base = Filters(macro_categories=filters.macro_categories, date_from=filters.date_from,
                   date_to=filters.date_to, states=filters.states)
    sub = meta[base.mask(meta)]
    if sub.empty:
        return {}
    bad = sub[sub["review_score"] <= 3]
    cats = bad["pred_categories"].fillna("").str.split("|").explode()
    cats = cats[cats != ""]
    top = (cats.value_counts() / max(len(bad), 1)).head(5)
    late = sub["is_late"].astype("boolean")
    return {
        "avaliacoes_com_texto_no_recorte": int(len(sub)),
        "nota_media": round(float(sub["review_score"].mean()), 2),
        "pct_nota_1_ou_2": round(float((sub["review_score"] <= 2).mean() * 100), 1),
        "pct_entregues_com_atraso": round(float(late.mean() * 100), 1) if late.notna().any() else None,
        "avaliacoes_com_problema": int(len(bad)),
        "pct_categoria_entre_problemas": {
            CATEGORIES[c].split(" (")[0]: round(float(v * 100), 1) for c, v in top.items()
        },
    }


def _format_numbers(nums):
    if not nums:
        return "(nenhum)"
    lines = [
        f"- avaliações com texto no recorte: {nums['avaliacoes_com_texto_no_recorte']}",
        f"- nota média: {nums['nota_media']}",
        f"- % com nota 1 ou 2: {nums['pct_nota_1_ou_2']}%",
    ]
    if nums.get("pct_entregues_com_atraso") is not None:
        lines.append(f"- % entregues com atraso (entre essas avaliações): {nums['pct_entregues_com_atraso']}%")
    lines.append(f"- avaliações com nota <= 3: {nums['avaliacoes_com_problema']}; entre elas, categoria estimada pelo classificador:")
    for k, v in nums["pct_categoria_entre_problemas"].items():
        lines.append(f"    - {k}: {v}%")
    return "\n".join(lines)


def _format_reviews(hits):
    out = []
    for i, r in enumerate(hits.itertuples(), start=1):
        date = pd.Timestamp(r.review_creation_date).strftime("%Y-%m")
        out.append(f"[R{i}] (nota {r.review_score}, {r.product_category}, {date}) {r.review_text}")
    return "\n".join(out)


def build_prompt(question, hits, numbers):
    return (
        f"PERGUNTA: {question}\n\n"
        f"NÚMEROS (calculados sobre todas as avaliações do recorte):\n{_format_numbers(numbers)}\n\n"
        f"AVALIAÇÕES RECUPERADAS:\n{_format_reviews(hits)}\n\n"
        "Resposta:"
    )


_CITE = re.compile(r"\[R(\d+)\]")


def cited_ids(answer, n_hits):
    nums = sorted({int(x) for x in _CITE.findall(answer)})
    valid = [n for n in nums if 1 <= n <= n_hits]
    invalid = [n for n in nums if n not in valid]
    return valid, invalid


def fallback_summary(hits, numbers):
    """Resumo sem LLM: agrupa as avaliações recuperadas pela categoria prevista."""
    if hits.empty:
        return "Nenhuma avaliação encontrada com esses filtros."
    lines = [f"As {len(hits)} avaliações mais parecidas com a pergunta, agrupadas pela categoria que o classificador previu:"]
    groups = {}
    for i, cats in enumerate(hits["pred_categories"].fillna(""), start=1):
        for c in (cats.split("|") if cats else ["sem_categoria"]):
            groups.setdefault(c, []).append(i)
    for cat, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        label = CATEGORIES.get(cat, "Sem categoria de reclamação")
        refs = ", ".join(f"[R{i}]" for i in ids)
        lines.append(f"- {label}: {len(ids)} avaliações {refs}")
    if numbers:
        n = f"{numbers['avaliacoes_com_texto_no_recorte']:,}".replace(",", ".")
        p = f"{numbers['pct_nota_1_ou_2']:.1f}".replace(".", ",")
        lines.append(f"\nNo recorte inteiro ({n} avaliações com texto), {p}% têm nota 1 ou 2.")
    return "\n".join(lines)


def answer(question, index, client=None, k=8, filters=None):
    filters = filters or filters_from_question(question)
    mask = filters.mask(index.meta)
    query = search_query(question, filters)
    hits = index.search(query, k=k, mask=mask)
    numbers = subset_numbers(index.meta, filters)

    if client is None:
        text = fallback_summary(hits, numbers)
        used_llm = False
    else:
        text = client.complete(RAG_SYSTEM, build_prompt(question, hits, numbers), max_tokens=4000)
        used_llm = True

    valid, invalid = cited_ids(text, len(hits))
    return {
        "question": question,
        "answer": text,
        "used_llm": used_llm,
        "filters": filters,
        "search_query": query,
        "numbers": numbers,
        "hits": hits,
        "cited": valid,
        "invalid_citations": invalid,
    }
