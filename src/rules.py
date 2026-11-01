"""Regras de palavra-chave para rotular categorias de reclamação.

O dataset não tem rótulo de categoria, então a abordagem é supervisão
fraca: estas regras rotulam as ~42 mil avaliações (com ruído) e o TF-IDF +
regressão logística aprende a partir delas. O experimento mede se o modelo
generaliza além das regras (resultado no README).

As regras rodam em cima do texto normalizado (minúsculo, sem acento).
"""
import re

from src.text import normalize

# "nao atrasou", "sem atraso", "nao houve muito atraso" -> não é reclamação.
# Esses trechos são removidos antes da busca por palavras de atraso.
_NEGATED_DELAY = re.compile(r"\b(nao|sem|nenhum|nenhuma)\s+(\w+\s+){0,2}(atras\w*|demor\w*)")
_GOOD_DEADLINE = re.compile(r"\b(antes|dentro) do prazo\b|\bno prazo\b")

PATTERNS = {
    "delivery": [
        r"\bnao (me )?(foi )?(entreg|cheg|receb)",
        r"\bainda nao\b",
        r"\b(atras|demor)\w*",
        r"\b(aguardando|aguardo|esperando) (o |meu |a |minha )?(produto|pedido|entrega|encomenda|mercadoria|recebimento)",
        r"\bcade\b",
        r"\b(extravi|roubad|perderam|perdid)\w*",
        r"\b(fora do|passou o|venceu o|apos o|depois do) prazo",
        r"\bprazo (vencido|expirado|nao (foi )?cumprido)",
        r"\bretir\w* (nos|no|na) (correio|agencia|central)",
        r"\brastre\w*",
        r"\bentreg\w* (a|para|pra) outra pessoa",
    ],
    "product_quality": [
        r"\b(defeit|quebrad|quebrou|danific|avari|amassad|riscad|arranh|rasgad|trincad|furad|vazand|vazament|estragad|mofad|embolorad|manchad|descascad)\w*",
        r"\bnao (esta |estao )?(funciona|funcionou|liga|ligou|carrega|pega)\b",
        r"\bparou de funcionar",
        r"\b(qualidade|material) (muito )?(ruim|baixa|pessima|inferior|fraca|fraco|duvidosa)",
        r"\b(baixa|pessima|ma|horrivel) qualidade",
        r"\bmal acabad|\bacabamento (ruim|pessimo)",
        r"\b(falsificad\w*|falso|falsos|falsa|falsas|pirata|replica)\b",
        r"\bnao (e|eh|era) original",
        r"\binferior\b",
    ],
    "wrong_product": [
        r"\b(produto|item|modelo|cor|tamanho|numero|voltagem) (errad|trocad|diferente)",
        r"\b(veio|recebi|enviaram|mandaram|entregaram|chegou) (o |um |uma )?(produto |item |modelo )?(errad|outr[oa]|diferente|trocad)",
        r"\bdiferente (do|da|de) (que|anunci|foto|imagem|descri|pedid|comprad|site)",
        r"\bnao (e|era|eh|foi) o (que|produto|modelo|mesmo)",
        r"\bnao (condiz|corresponde|confere|conferi)",
        r"\b(em vez|ao inves|no lugar) d",
        r"\bpropaganda enganosa",
        r"\bnao (e|era) compativel|\bnao serve\b",
        r"\bnao (estava|esta|veio) (de acordo|conforme)",
    ],
    "missing_items": [
        r"\bfalt(ou|ando|a|aram|am|ante)\b(?! de (contato|respeito|informac|compromisso|atenc|educac|resposta))",
        r"\b(so|somente|apenas) (veio|chegou|recebi|entregaram|foi entregue|chegaram|vieram) (1|um|uma|01|parte|metade|2|dois|duas)\b",
        r"\b(recebi|veio|chegou|chegaram|entregaram) (so|somente|apenas) (1|um|uma|01|parte|metade|2|dois|duas)\b",
        r"\bincomplet",
        r"\bnao (veio|vieram|vem|acompanha) (o|a|os|as|com) ",
        r"\b(um|uma|1) dos (itens|produtos)",
        r"\bparte do (pedido|produto|material|compra)",
    ],
    "customer_service": [
        r"\b(sem|nenhum\w*|nao (tive|obtive|recebi|tenho)|aguardando|esperando|aguardo) (\w+ ){0,2}(resposta|retorno|contato|satisfacao|solucao|posicionamento)",
        r"\bnao (me )?(respond|retorn|resolv|atend|deram)\w*",
        r"\bninguem (respond|resolv|retorn|atend)\w*",
        r"\b(reembolso|estorno|ressarcimento|dinheiro de volta)",
        r"\bcancel\w*",
        r"\b(devolu\w*|devolver|troca|trocar)\b",
        r"\bestoque\b",
        r"\b(sem|nao (veio|recebi|emitiu|enviaram|mandaram)) (com )?(a )?(nota fiscal|nf)\b",
        r"\b(procon|reclame aqui|descaso|desrespeito)\b",
        r"\bfalta de (contato|respeito|compromisso|informac)\w*",
        r"\b(cobrad|cobranca|cobraram)\w*",
    ],
}

COMPILED = {cat: [re.compile(p) for p in pats] for cat, pats in PATTERNS.items()}


def _prepare(text_norm):
    text = _NEGATED_DELAY.sub(" ", text_norm)
    return text


def label_categories(text, already_normalized=False):
    t = text if already_normalized else normalize(text)
    t = _prepare(t)
    cats = []
    for cat, pats in COMPILED.items():
        if any(p.search(t) for p in pats):
            cats.append(cat)
    # "chegou antes do prazo, mas atrasou a nota" etc. é raro; o caso comum
    # é "chegou no prazo" disparando delivery por causa de "ainda nao usei".
    if "delivery" in cats and _GOOD_DEADLINE.search(t) and not re.search(r"\bnao (me )?(foi )?(entreg|cheg|receb)|atras|demor", t):
        cats.remove("delivery")
    return cats


# --- severidade ----------------------------------------------------------------

_NOT_RECEIVED = re.compile(
    r"\bnao (me )?(foi )?(entreg|cheg|receb)|\bainda nao (recebi|chegou)|\b(aguardando|aguardo|esperando) (o |meu |a |minha )?(produto|pedido|entrega|encomenda|mercadoria|recebimento)|\bcade\b|\bextravi|\broubad"
)
_BROKEN = re.compile(
    r"\b(quebrad|quebrou|defeit|danific|avariad|trincad|vazand|estragad|mofad|embolorad)\w*|\bnao (esta |estao )?(funciona|funcionou|liga|ligou|carrega)\b|\bparou de funcionar|\bfalsific|\bfalso"
)
_ESCALATION = re.compile(r"\b(procon|reclame aqui|golpe|fraude|lesad|processo|justica)\w*")
_MILD = re.compile(r"\b(pouco|pequen|leve|levemente|so um detalhe|unico detalhe)\w*")


def severity_from_rules(text, categories, sentiment, already_normalized=False):
    """Severidade aproximada para quando não tem LLM.

    Aproximação: não entende contexto, apenas combina sinais.
    """
    t = text if already_normalized else normalize(text)
    if not categories:
        if sentiment == "negative":
            return "low"
        return "none"
    if _ESCALATION.search(t):
        return "high"
    if "missing_items" in categories or "wrong_product" in categories:
        return "high" if sentiment == "negative" else "low"
    if "delivery" in categories and _NOT_RECEIVED.search(t):
        return "high"
    if "product_quality" in categories and _BROKEN.search(t):
        return "high"
    if sentiment == "positive" or _MILD.search(t):
        return "low"
    return "medium"
