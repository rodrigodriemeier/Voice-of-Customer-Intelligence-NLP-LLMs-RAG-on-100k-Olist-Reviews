"""Indicadores do dashboard, em SQL (DuckDB) em cima dos parquets.

Todo número que aparece no dashboard sai daqui. Nada é estimado por LLM.

As categorias de reclamação vêm do classificador TF-IDF (pred_categories),
então carregam o erro dele (macro-F1 ~0,70 no gold). O dashboard avisa isso.
"""
import duckdb
import pandas as pd

from src.config import CATEGORIES, DATA_PROCESSED

REVIEWS = DATA_PROCESSED / "reviews.parquet"
SCORED = DATA_PROCESSED / "corpus_scored.parquet"


def _where(filters):
    """Monta o WHERE a partir do dict de filtros da tela."""
    clauses, params = ["1=1"], []
    if filters.get("date_from"):
        clauses.append("review_creation_date >= ?")
        params.append(pd.Timestamp(filters["date_from"]))
    if filters.get("date_to"):
        clauses.append("review_creation_date <= ?")
        params.append(pd.Timestamp(filters["date_to"]))
    if filters.get("macro"):
        clauses.append(f"macro_category IN ({','.join('?' * len(filters['macro']))})")
        params.extend(filters["macro"])
    if filters.get("states"):
        clauses.append(f"customer_state IN ({','.join('?' * len(filters['states']))})")
        params.extend(filters["states"])
    return " AND ".join(clauses), params


class Analytics:
    def __init__(self, reviews_path=REVIEWS, scored_path=SCORED):
        self.con = duckdb.connect()
        self.con.execute(f"CREATE VIEW reviews AS SELECT * FROM read_parquet('{reviews_path.as_posix()}')")
        self.con.execute(f"CREATE VIEW scored AS SELECT * FROM read_parquet('{scored_path.as_posix()}')")

    def q(self, sql, params=None):
        return self.con.execute(sql, params or []).df()

    def options(self):
        macros = self.q("SELECT DISTINCT macro_category FROM reviews WHERE macro_category IS NOT NULL ORDER BY 1")
        states = self.q("SELECT DISTINCT customer_state FROM reviews WHERE customer_state IS NOT NULL ORDER BY 1")
        dates = self.q("SELECT min(review_creation_date) AS a, max(review_creation_date) AS b FROM reviews")
        return macros.iloc[:, 0].tolist(), states.iloc[:, 0].tolist(), (dates.a[0], dates.b[0])

    def kpis(self, filters):
        w, p = _where(filters)
        return self.q(f"""
            SELECT count(*)                                       AS avaliacoes,
                   avg(review_score)                              AS nota_media,
                   avg(CASE WHEN review_score <= 2 THEN 1.0 ELSE 0 END) AS pct_negativas,
                   avg(CASE WHEN is_late THEN 1.0 WHEN is_late = false THEN 0 END) AS pct_atraso,
                   avg(CASE WHEN has_text THEN 1.0 ELSE 0 END)     AS pct_com_texto
            FROM reviews WHERE {w}
        """, p).iloc[0].to_dict()

    def negative_rate_by(self, filters, level="macro_category", min_reviews=300):
        w, p = _where(filters)
        return self.q(f"""
            SELECT {level} AS grupo,
                   count(*) AS avaliacoes,
                   avg(CASE WHEN review_score <= 2 THEN 1.0 ELSE 0 END) AS pct_negativas
            FROM reviews
            WHERE {w} AND {level} IS NOT NULL
            GROUP BY 1 HAVING count(*) >= {int(min_reviews)}
            ORDER BY pct_negativas DESC
        """, p)

    def complaint_causes(self, filters):
        """Entre as avaliações com texto e nota <= 3, quanto cada categoria aparece."""
        w, p = _where(filters)
        df = self.q(f"""
            WITH bad AS (SELECT * FROM scored WHERE {w} AND review_score <= 3),
                 exploded AS (SELECT unnest(string_split(pred_categories, '|')) AS cat FROM bad)
            SELECT cat, count(*) AS n, count(*) * 1.0 / (SELECT count(*) FROM bad) AS pct
            FROM exploded WHERE cat <> '' GROUP BY 1 ORDER BY n DESC
        """, p)
        df["categoria"] = df["cat"].map(lambda c: CATEGORIES.get(c, c).split(" (")[0])
        return df

    def delay_vs_negative(self, filters):
        w, p = _where(filters)
        order = ["7+ dias adiantado", "0-6 dias adiantado", "1-3 dias atrasado", "4-7 dias atrasado",
                 "8-14 dias atrasado", "15+ dias atrasado", "não entregue"]
        df = self.q(f"""
            SELECT delay_bucket AS faixa, count(*) AS avaliacoes,
                   avg(CASE WHEN review_score <= 2 THEN 1.0 ELSE 0 END) AS pct_negativas,
                   avg(review_score) AS nota_media
            FROM reviews WHERE {w} AND delay_bucket IS NOT NULL GROUP BY 1
        """, p)
        df["faixa"] = pd.Categorical(df["faixa"], categories=order, ordered=True)
        return df.sort_values("faixa")

    def monthly(self, filters):
        """Evolução mensal, de jan/2017 a ago/2018: fora desse intervalo há
        poucas avaliações por mês e as taxas ficam instáveis."""
        w, p = _where(filters)
        base = self.q(f"""
            SELECT date_trunc('month', review_creation_date) AS mes, count(*) AS avaliacoes,
                   avg(CASE WHEN review_score <= 2 THEN 1.0 ELSE 0 END) AS pct_negativas
            FROM reviews WHERE {w} GROUP BY 1 ORDER BY 1
        """, p)
        cats = self.q(f"""
            WITH m AS (
                SELECT date_trunc('month', review_creation_date) AS mes, pred_categories
                FROM scored WHERE {w}
            ), tot AS (SELECT mes, count(*) AS n FROM m GROUP BY 1),
            ex AS (SELECT mes, unnest(string_split(pred_categories, '|')) AS cat FROM m)
            SELECT ex.mes, cat, count(*) * 1.0 / any_value(tot.n) AS pct
            FROM ex JOIN tot USING (mes) WHERE cat <> '' GROUP BY 1, 2 ORDER BY 1
        """, p)
        cats["categoria"] = cats["cat"].map(lambda c: CATEGORIES.get(c, c).split(" (")[0])
        lo, hi = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-08-31")
        return base[base.mes.between(lo, hi)], cats[cats.mes.between(lo, hi)]

    def examples(self, filters, category, n=5):
        w, p = _where(filters)
        return self.q(f"""
            SELECT review_score, product_category, review_text
            FROM scored
            WHERE {w} AND review_score <= 3 AND list_contains(string_split(pred_categories, '|'), ?)
            ORDER BY random() LIMIT {int(n)}
        """, p + [category])
