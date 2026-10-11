"""Serves the model's predictions to the API.

It only reads the artifacts written by `python -m data_science.predict` (outputs/); it contains no training code and
never touches the raw data, so the API runs without the Kaggle dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"   # optional: <article_id>.jpg files, served at /images

FEATURE_LABELS = {
    "score_lag1": "Success score last quarter",
    "score_lag2": "Success score 2 quarters ago",
    "score_lag3": "Success score 3 quarters ago",
    "score_lag4": "Success score in the same quarter last year",
    "score_mean4": "Average success score over the last 4 quarters",
    "score_max4": "Best quarter of the last 4",
    "score_trend4": "Trend over the last 4 quarters",
    "quarters_sold4": "Quarters with sales (of the last 4)",
    "age_quarters": "Quarters since the style first sold",
    "quarters_since_last_sale": "Quarters since the last sale",
    "log_revenue_lag1": "Revenue last quarter",
    "log_units_lag1": "Units sold last quarter",
    "articles_lag1": "Articles sold last quarter",
    "full_price_factor_lag1": "Share of full price achieved last quarter",
    "score_last4w": "Success score in the last 4 weeks",
    "revenue_momentum": "Revenue momentum (last 4 weeks vs last quarter)",
    "new_articles_13w": "New articles launched in the last 13 weeks",
    "style_season_share": "Share of the style's past revenue earned in this season",
    "category_season_share": "Share of the category's past revenue earned in this season",
    "colour_momentum": "Colour gaining share across all categories",
    "pattern_momentum": "Pattern gaining share across all categories",
    "tone_momentum": "Colour tone gaining share across all categories",
    "category_growth": "Category revenue growth year on year",
    "category_n_styles_lag1": "Competing styles in the category",
    "category_benchmark_lag1": "Category benchmark",
    "log_customers_lag1": "Buyers last quarter",
    "repeat_rate4": "Repeat-purchase rate (last year)",
    "online_share_lag1": "Online share of units last quarter",
    "price_index_lag1": "Price relative to the category",
    "index_group_name": "Customer segment",
    "product_type_name": "Product type",
    "graphical_appearance_name": "Pattern",
    "colour_group_name": "Colour",
    "perceived_colour_value_name": "Colour tone",
    "target_season": "Season",
}
LOG_FEATURES = {"log_revenue_lag1", "log_units_lag1", "log_customers_lag1"}   # shown on their natural scale


class StyleNotFound(KeyError):
    pass


def _display_value(feature: str, raw: str) -> str:
    try:
        value = float(raw)
    except ValueError:
        return raw                                   # categorical feature
    if np.isnan(value):
        return "no history"
    if feature in LOG_FEATURES:
        value = np.expm1(value)
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2f}"


def _rows(df: pd.DataFrame, style_id: str) -> pd.DataFrame | None:
    """Rows of one style from a frame indexed (and sorted) by style_id; None if the style has none."""
    try:
        loc = df.index.get_loc(style_id)        # a slice on a sorted index, so this is a cheap view
    except KeyError:
        return None
    return df.iloc[[loc]] if isinstance(loc, int) else df.iloc[loc]


def quarter_label(quarter_id: str) -> str:
    """'2020_autumn' → 'Autumn 2020'."""
    year, season = quarter_id.split("_")
    return f"{season.title()} {year}"


def _article_id(article_id: int) -> str:
    return f"{int(article_id):010d}"


class StyleService:
    def __init__(self, outputs_dir: Path = OUTPUTS_DIR, images_dir: Path = IMAGES_DIR):
        self.images_dir = images_dir
        pred = pd.read_csv(outputs_dir / "predictions.csv")
        pred["rank"] = pred["rank"].astype("Int64")
        self.predictions = pred.set_index("style_id", drop=False)
        self.quarter = str(pred.quarter_id.iloc[0])
        self.ranked = pred[pred.commercially_relevant].sort_values("rank")
        self.top_score = float(self.ranked.prediction_score.iloc[0])

        # indexed by style_id (rows in their stored order), so each lookup is a cheap index slice
        self.history = pd.read_parquet(outputs_dir / "style_history.parquet").sort_values(["style_id", "start"]).set_index("style_id")
        self.explanations = pd.read_parquet(outputs_dir / "explanations.parquet").set_index("style_id").sort_index(kind="stable")
        self.articles = pd.read_parquet(outputs_dir / "style_articles.parquet").set_index("style_id").sort_index(kind="stable")

    # ------------------------------------------------------------ helpers
    def _image_url(self, article_id: int) -> str | None:
        aid = _article_id(article_id)
        return f"/images/{aid}.jpg" if (self.images_dir / f"{aid}.jpg").exists() else None

    def _sales_history(self, style_id: str) -> list[dict]:
        h = _rows(self.history, style_id)
        if h is None:
            return []
        return [{"quarter_id": r.quarter_id, "season": r.season, "success_score": round(float(r.success_score), 3),
                 "revenue": round(float(r.revenue), 3), "units": int(r.units), "articles": int(r.articles)}
                for r in h.itertuples()]

    def _summary(self, row: pd.Series) -> dict:
        history = self._sales_history(row.style_id)
        articles = _rows(self.articles, row.style_id)
        best = articles.iloc[0] if articles is not None and len(articles) else None
        return {
            "rank": None if pd.isna(row["rank"]) else int(row["rank"]),
            "style_id": row.style_id,
            "name": row.style_name,
            "category": row.group_name,
            "prediction_score": round(float(row.prediction_score), 3),
            "relative_score": round(float(row.prediction_score) / self.top_score, 3),
            "percentile": None if pd.isna(row["rank"]) else round((len(self.ranked) - int(row["rank"])) / (len(self.ranked) - 1), 4),
            "units_last_quarter": history[-1]["units"] if history else 0,
            "image_url": None if best is None else self._image_url(best.article_id),
            "sales_history": history,
        }

    def _explanation(self, row: pd.Series) -> dict:
        e = _rows(self.explanations, row.style_id)
        contributions = [] if e is None else [
            {"feature": r.feature, "label": FEATURE_LABELS.get(r.feature, r.feature),
             "value": _display_value(r.feature, r.feature_value), "contribution": round(float(r.contribution), 3),
             "effect": "raises" if r.contribution >= 0 else "lowers"}
            for r in e.itertuples()]
        drivers = [c for c in contributions if c["effect"] == "raises"][:2]
        position = (f"Ranked #{int(row['rank'])} of {len(self.ranked):,} commercially relevant styles"
                    if not pd.isna(row["rank"]) else "Not eligible for the ranking (revenue last quarter below the top 10%)")
        reasons = "; ".join(f"{c['label'].lower()}: {c['value']}" for c in drivers)
        summary = f"{position} for {quarter_label(self.quarter)}." + (f" Main drivers: {reasons}." if reasons else "")
        return {"summary": summary, "contributions": contributions}

    # ------------------------------------------------------------ public
    @property
    def n_styles(self) -> int:
        return len(self.predictions)

    def top(self, limit: int) -> dict:
        styles = [self._summary(r) for _, r in self.ranked.head(limit).iterrows()]
        return {"quarter": self.quarter, "candidates": len(self.ranked), "styles": styles}

    def list(self, limit: int, offset: int, segment: str | None = None, product_type: str | None = None,
             include_all: bool = False) -> dict:
        rows = self.predictions.sort_values("rank_all_styles") if include_all else self.ranked
        if segment:
            rows = rows[rows.index_group_name.str.lower() == segment.lower()]
        if product_type:
            rows = rows[rows.product_type_name.str.lower() == product_type.lower()]
        page = rows.iloc[offset:offset + limit]
        return {"quarter": self.quarter, "total": len(rows), "limit": limit, "offset": offset,
                "styles": [self._summary(r) for _, r in page.iterrows()]}

    def get(self, style_id: str) -> dict:
        if style_id not in self.predictions.index:
            raise StyleNotFound(style_id)
        row = self.predictions.loc[style_id]
        articles = _rows(self.articles, style_id)
        articles = [] if articles is None else [
            {"article_id": _article_id(a.article_id), "name": a.prod_name,
             "description": None if pd.isna(a.detail_desc) else a.detail_desc,
             "units_last_quarter": int(a.units_last_quarter), "image_url": self._image_url(a.article_id),
             "kaggle_image_path": f"images/{_article_id(a.article_id)[:3]}/{_article_id(a.article_id)}.jpg"}
            for a in articles.itertuples()]
        return {
            **{k: v for k, v in self._summary(row).items() if k not in ("units_last_quarter", "image_url")},
            "quarter": self.quarter,
            "commercially_relevant": bool(row.commercially_relevant),
            "attributes": {"segment": row.index_group_name, "product_type": row.product_type_name,
                           "pattern": row.graphical_appearance_name, "colour": row.colour_group_name,
                           "tone": row.perceived_colour_value_name},
            "articles": articles,
            "explanation": self._explanation(row),
        }
