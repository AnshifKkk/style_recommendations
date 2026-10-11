"""Serves the model's predictions to the API.

It only reads the artifacts written by `python -m data_science.predict` (outputs/); it contains no training code and
never touches the raw data, so the API runs without the Kaggle dataset.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from backend.images import OnDemandImages
from data_science.features.labels import FEATURE_LABELS, LOG_FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"   # optional: <style_id>/<article_id>.jpg files, served at /images
# Where images come from. Unset: local files in IMAGES_DIR (fetched by `python -m data_science.fetch_images`).
# Set it to point at an image store / CDN instead, e.g. "https://cdn.example.com/images/{prefix}/{article_id}.jpg"
# ({article_id} = 10 digits, {prefix} = its first 3 digits). Changing the source needs no code change.
IMAGE_URL_TEMPLATE = os.environ.get("IMAGE_URL_TEMPLATE")
# Local mode: fetch a missing image from Kaggle the first time the app shows it (set IMAGES_ON_DEMAND=0 to disable)
IMAGES_ON_DEMAND = os.environ.get("IMAGES_ON_DEMAND", "1") != "0"
CONCEPTS_DIR = OUTPUTS_DIR / "concepts"   # optional: <style_id>.png + concepts.json from the concept-generation step
CONCEPTS_IMAGE = OUTPUTS_DIR / "generated_concepts.png"   # optional: all three concepts in one image
COMING_SOON_QUARTERS = 4                  # seasons after the prediction quarter shown as "coming soon"
SEASON_START_MONTH = {"spring": 3, "summer": 6, "autumn": 9, "winter": 12}


class StyleNotFound(KeyError):
    pass


class SeasonNotFound(KeyError):
    pass


def _quarter_dates(quarter_id: str) -> tuple[str, str]:
    year, season = quarter_id.split("_")
    start = pd.Timestamp(year=int(year), month=SEASON_START_MONTH[season], day=1)
    return start.date().isoformat(), (start + pd.DateOffset(months=3) - pd.Timedelta(days=1)).date().isoformat()


def _next_quarter(quarter_id: str) -> str:
    year, season = quarter_id.split("_")
    seasons = list(SEASON_START_MONTH)
    return f"{int(year) + 1}_spring" if season == "winter" else f"{year}_{seasons[seasons.index(season) + 1]}"


def _category_phrase(group_name: str) -> str:
    """'Ladieswear · Skirt' → 'Ladieswear skirt'."""
    segment, product_type = group_name.split(" · ", 1)
    return f"{segment} {product_type.lower()}"


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
    def __init__(self, outputs_dir: Path = OUTPUTS_DIR, images_dir: Path = IMAGES_DIR,
                 image_url_template: str | None = IMAGE_URL_TEMPLATE, concepts_dir: Path = CONCEPTS_DIR,
                 images_on_demand: bool = IMAGES_ON_DEMAND):
        self.images_dir = images_dir
        self.image_url_template = image_url_template
        self.image_store = OnDemandImages(images_dir, enabled=images_on_demand)
        self.concepts_dir = concepts_dir
        concepts_file = concepts_dir / "concepts.json"
        self.concepts = pd.read_json(concepts_file, typ="series").to_dict() if concepts_file.exists() else {}
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
        self.winners = pd.read_parquet(outputs_dir / "season_winners.parquet")
        self.season_articles = pd.read_parquet(outputs_dir / "season_articles.parquet")
        self.best_actual_score = float(self.winners.loc[self.winners.status == "actual", "success_score"].max())
        # the only images the app can show (and so the only ones the API will fetch on demand)
        pairs = pd.concat([self.articles.reset_index()[["style_id", "article_id"]],
                           self.season_articles[["style_id", "article_id"]]])
        self.known_images = {(s, _article_id(a)) for s, a in zip(pairs.style_id, pairs.article_id)}

    # ------------------------------------------------------------ helpers
    def _image_url(self, style_id: str, article_id: int) -> str | None:
        aid = _article_id(article_id)
        if self.image_url_template:
            return self.image_url_template.format(article_id=aid, prefix=aid[:3])
        if self.image_store.enabled or self.image_store.path(style_id, aid).exists():
            return f"/images/{style_id}/{aid}.jpg"     # served (and fetched if missing) by GET /images/...
        return None

    def image_file(self, style_id: str, article_id: str) -> Path | None:
        """Local file of a known article image, fetched first if missing; None if unknown or unavailable."""
        if (style_id, article_id) not in self.known_images:
            return None
        return self.image_store.get(style_id, article_id)

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
            "image_url": None if best is None else self._image_url(row.style_id, best.article_id),
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
             "units_last_quarter": int(a.units_last_quarter), "image_url": self._image_url(style_id, a.article_id),
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

    # ------------------------------------------------------------ seasons (the app's timeline)
    def seasons(self) -> list[dict]:
        known = self.winners.drop_duplicates("quarter_id").sort_values("start")
        out = [self._season_info(r.quarter_id, r.status) for r in known.itertuples()]
        upcoming = self.quarter
        for _ in range(COMING_SOON_QUARTERS):
            upcoming = _next_quarter(upcoming)
            out.append(self._season_info(upcoming, "coming_soon"))
        return out

    def season(self, quarter_id: str) -> dict:
        statuses = {s["quarter_id"]: s["status"] for s in self.seasons()}
        if quarter_id not in statuses:
            raise SeasonNotFound(quarter_id)
        info = self._season_info(quarter_id, statuses[quarter_id])
        if info["status"] == "coming_soon":
            return {**info, "styles": []}
        winners = self.winners[self.winners.quarter_id == quarter_id].sort_values("rank")
        articles = self.season_articles[self.season_articles.quarter_id == quarter_id]
        styles = [self._season_style(w, articles[articles.style_id == w.style_id], info) for w in winners.itertuples()]
        concepts_image = "/concepts-image" if info["status"] == "predicted" and CONCEPTS_IMAGE.exists() else None
        return {**info, "styles": styles, "concepts_image_url": concepts_image}

    def _season_info(self, quarter_id: str, status: str) -> dict:
        start, end = _quarter_dates(quarter_id)
        return {"quarter_id": quarter_id, "label": quarter_label(quarter_id), "status": status, "start": start, "end": end}

    def _season_style(self, w, articles: pd.DataFrame, info: dict) -> dict:
        score, category = float(w.success_score), _category_phrase(w.group_name)
        if info["status"] == "predicted":
            explanation = (f"{score:.2f} (predicted): expected to earn about {score:.1f}× as much as a typical {category} "
                           f"style in {info['label']}, on revenue adjusted for discounts. The best score seen in past "
                           f"seasons is {self.best_actual_score:.2f}, so read this as very strong: the model is validated "
                           "on the ranking, not on the exact value.")
        else:
            explanation = (f"{score:.2f}: this style earned about {score:.1f}× as much as a typical {category} style in "
                           f"{info['label']}, on revenue adjusted for discounts (1.0 = a typical {category} style).")
        concept = None
        if info["status"] == "predicted":
            image = self.concepts_dir / f"{w.style_id}.png"
            concept = {"image_url": f"/concepts/{w.style_id}.png" if image.exists() else None,
                       "description": self.concepts.get(w.style_id)}
        return {
            "rank": int(w.rank), "style_id": w.style_id, "name": w.style_name, "category": w.group_name,
            "success_score": round(score, 3), "score_explanation": explanation, "concept": concept,
            "articles": [{"article_id": _article_id(a.article_id), "name": a.prod_name,
                          "description": None if pd.isna(a.detail_desc) else a.detail_desc, "units": int(a.units),
                          "image_url": self._image_url(w.style_id, a.article_id)}
                         for a in articles.sort_values("units", ascending=False).itertuples()],
        }
