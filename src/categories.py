"""Agrupamento das 73 categorias de produto da Olist em macrocategorias.

Mapeamento manual. Algumas escolhas são discutíveis (relógios em "Moda",
cool_stuff em "Outros"); o objetivo é ter poucos grupos coerentes para o
painel e para os filtros do RAG.
"""

MACRO = {
    "Eletrônicos e informática": [
        "informatica_acessorios", "telefonia", "eletronicos", "consoles_games",
        "audio", "telefonia_fixa", "pcs", "tablets_impressao_imagem", "cine_foto",
        "pc_gamer",
    ],
    "Eletrodomésticos": [
        "eletrodomesticos", "eletrodomesticos_2", "eletroportateis", "climatizacao",
        "portateis_casa_forno_e_cafe", "portateis_cozinha_e_preparadores_de_alimentos",
    ],
    "Casa e decoração": [
        "cama_mesa_banho", "moveis_decoracao", "utilidades_domesticas",
        "moveis_escritorio", "moveis_sala", "casa_conforto", "casa_conforto_2",
        "moveis_cozinha_area_de_servico_jantar_e_jardim", "moveis_quarto",
        "moveis_colchao_e_estofado", "la_cuisine", "artigos_de_natal", "flores",
    ],
    "Beleza e saúde": ["beleza_saude", "perfumaria", "fraldas_higiene"],
    "Moda e acessórios": [
        "relogios_presentes", "fashion_bolsas_e_acessorios", "malas_acessorios",
        "fashion_calcados", "fashion_roupa_masculina", "fashion_underwear_e_moda_praia",
        "fashion_roupa_feminina", "fashion_esporte", "fashion_roupa_infanto_juvenil",
    ],
    "Esporte e lazer": ["esporte_lazer", "instrumentos_musicais", "artigos_de_festas"],
    "Bebês e brinquedos": ["bebes", "brinquedos"],
    "Ferramentas e construção": [
        "ferramentas_jardim", "construcao_ferramentas_construcao", "casa_construcao",
        "construcao_ferramentas_iluminacao", "construcao_ferramentas_jardim",
        "sinalizacao_e_seguranca", "construcao_ferramentas_seguranca",
        "construcao_ferramentas_ferramentas",
    ],
    "Automotivo": ["automotivo"],
    "Livros, papelaria e mídia": [
        "papelaria", "livros_interesse_geral", "livros_tecnicos", "livros_importados",
        "dvds_blu_ray", "musica", "cds_dvds_musicais", "artes", "artes_e_artesanato",
    ],
    "Alimentos e bebidas": ["alimentos", "bebidas", "alimentos_bebidas"],
    "Pet shop": ["pet_shop"],
    "Outros": [
        "cool_stuff", "market_place", "industria_comercio_e_negocios",
        "agro_industria_e_comercio", "seguros_e_servicos",
    ],
}

CATEGORY_TO_MACRO = {cat: macro for macro, cats in MACRO.items() for cat in cats}

# Palavras que, se aparecerem numa pergunta do usuário, ativam o filtro de
# macrocategoria no RAG. Sem acento porque o texto é normalizado antes.
MACRO_KEYWORDS = {
    "Eletrônicos e informática": [
        "eletronic", "informatica", "celular", "smartphone", "telefone", "fone",
        "notebook", "computador", "pc ", "videogame", "console", "tablet", "camera",
    ],
    "Eletrodomésticos": ["eletrodomestic", "liquidificador", "ventilador", "ar condicionado", "cafeteira"],
    "Casa e decoração": ["moveis", "movel", "decoracao", "cama", "lencol", "toalha", "cozinha"],
    "Beleza e saúde": ["beleza", "perfume", "cosmetic", "maquiagem", "saude"],
    "Moda e acessórios": ["roupa", "moda", "relogio", "bolsa", "calcado", "sapato", "tenis", "mala"],
    "Esporte e lazer": ["esporte", "academia", "bicicleta", "instrumento"],
    "Bebês e brinquedos": ["bebe", "brinquedo", "infantil"],
    "Ferramentas e construção": ["ferramenta", "construcao", "jardim", "iluminacao"],
    "Automotivo": ["automotiv", "carro", "moto "],
    "Livros, papelaria e mídia": ["livro", "papelaria", "caderno", "dvd"],
    "Alimentos e bebidas": ["alimento", "bebida", "comida"],
    "Pet shop": ["pet", "cachorro", "gato", "racao"],
}


def to_macro(category):
    if not isinstance(category, str) or not category:
        return "Sem categoria"
    return CATEGORY_TO_MACRO.get(category, "Outros")


def pretty(category):
    """cama_mesa_banho -> 'Cama mesa banho'."""
    if not isinstance(category, str) or not category:
        return "Sem categoria"
    return category.replace("_", " ").capitalize()
