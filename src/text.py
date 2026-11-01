import re
import unicodedata

# A Olist anonimizou o nome das lojas trocando por casas de Game of Thrones
# ("comprei nas lojas lannister..."). Para o modelo isso é ruído: ele acaba
# aprendendo que "lannister" é sinal de alguma coisa. Os nomes viram "loja".
_GOT_STORES = re.compile(r"\b(lannister|stark|targaryen|baratheon|tyrell|martell|arryn|greyjoy)s?\b")
_SPACES = re.compile(r"\s+")


def clean_display(text):
    """Limpeza mínima para mostrar na tela: tira \\r\\n e espaços repetidos."""
    if not isinstance(text, str):
        return ""
    return _SPACES.sub(" ", text).strip()


def strip_accents(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize(text):
    """Versão para os modelos: minúsculo, sem acento, lojas anonimizadas."""
    text = strip_accents(clean_display(text).lower())
    text = _GOT_STORES.sub("loja", text)
    # pontuação vira espaço ("nao.chegou" -> "nao chegou")
    text = re.sub(r"[^\w\s]", " ", text)
    return _SPACES.sub(" ", text).strip()


def build_review_text(title, message):
    title = clean_display(title)
    message = clean_display(message)
    if title and message:
        # alguns títulos são só repetição do começo da mensagem
        if message.lower().startswith(title.lower()):
            return message
        sep = " " if title.endswith((".", "!", "?")) else ". "
        return f"{title}{sep}{message}"
    return title or message


# Stopwords curtas. "não"/"nao" fica de fora de propósito: em avaliação é uma
# das palavras mais informativas ("não chegou", "não funciona").
STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das", "em", "no",
    "na", "nos", "nas", "e", "que", "para", "pra", "por", "com", "se", "me", "eu",
    "meu", "minha", "ao", "aos", "foi", "ja", "esta", "isso", "este", "essa",
}
