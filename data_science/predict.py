"""Score every style for the prediction quarter and explain each prediction.

    python -m data_science.predict                 # predict, write the API artifacts, fetch the images
    python -m data_science.predict --no-images     # skip fetching images

Writes (these files are everything the backend API needs; it never reads raw data or training code)
- outputs/predictions.csv          one row per style: predicted success score, rank among commercially relevant styles
- outputs/explanations.parquet     the largest feature contributions per style
- outputs/style_history.parquet    success score, revenue, units per style per past quarter
- outputs/style_articles.parquet   each style's best-selling articles (name, description, units)
- outputs/season_winners.parquet   the top 3 styles of every season: actual for past quarters, predicted for the next
- outputs/season_articles.parquet  the 5 best-selling articles of each of those styles in that season
- outputs/images_needed.csv        the article images the app shows; missing ones are then fetched into outputs/images/

Explanations: the model is linear in log(1 + success score), so each prediction is exactly
    average prediction + Σ contribution(feature),   contribution = coefficient × (value − training average)
summed over the encoded columns of a feature. For a linear model this equals its SHAP values.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from data_science.data import load_articles, load_transactions
from data_science.features.build import CATEGORICAL_FEATURES, FEATURES, TARGET
from data_science.features.build import predict_quarter_after
from data_science.features.quarters import build_style_quarter_table, quarters_between
from data_science.features.style import STYLE_ATTRIBUTES, add_style_columns
from data_science.fetch_images import ImageSourceUnavailable, fetch
from data_science.train import (COMMERCIAL_TOP_SHARE, LAST_TRAINING_DAY, MODEL_PATH, OUTPUTS, as_model_input,
                                commercially_relevant, load_or_build_features)

TOP_CONTRIBUTIONS = 8
ARTICLES_PER_STYLE = 5
TOP_STYLES = 3      # the winning styles shown per season (with their ARTICLES_PER_STYLE best-selling articles)


def _original_feature(encoded_name: str) -> str:
    """'num__missingindicator_score_lag4' → 'score_lag4'; 'cat__product_type_name_Dress' → 'product_type_name'."""
    kind, name = encoded_name.split("__", 1)
    if kind == "num":
        return name.removeprefix("missingindicator_")
    return next(c for c in CATEGORICAL_FEATURES if name.startswith(c + "_"))


def explain(bundle: dict, rows: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Predictions (log scale) and per-feature contributions (rows × features)."""
    model, train_mean = bundle["model"], bundle["train_mean"]
    pre, ridge = model.named_steps["pre"], model.named_steps["ridge"]
    x = pre.transform(as_model_input(rows))
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    contrib = (x - train_mean) * ridge.coef_
    by_feature = pd.DataFrame(contrib, columns=[_original_feature(n) for n in pre.get_feature_names_out()])
    by_feature = by_feature.T.groupby(level=0).sum().T[FEATURES]
    by_feature.index = rows.index
    return model.predict(as_model_input(rows)), by_feature


def predict_quarter(bundle: dict, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranked predictions and the largest contributions for every style of the prediction quarter."""
    rows = features[features[TARGET].isna()].copy()
    log_pred, contrib = explain(bundle, rows)
    rows["prediction_score"] = np.expm1(log_pred)
    rows["commercially_relevant"] = commercially_relevant(rows)
    # rank among commercially relevant styles (the business rule); other styles get no rank
    rows["rank"] = rows.prediction_score.where(rows.commercially_relevant).rank(ascending=False, method="first").astype("Int64")
    rows["rank_all_styles"] = rows.prediction_score.rank(ascending=False, method="first").astype(int)
    pred = rows.sort_values("rank_all_styles")[["rank", "rank_all_styles", "commercially_relevant", "style_id", "style_name",
                                                "group_name", *STYLE_ATTRIBUTES, "quarter_id", "prediction_score",
                                                "score_lag1", "score_lag4"]]

    long = contrib.assign(style_id=rows.style_id).melt(id_vars="style_id", var_name="feature", value_name="contribution")
    values = rows[["style_id", *FEATURES]].astype({c: str for c in FEATURES}).melt(
        id_vars="style_id", var_name="feature", value_name="feature_value")
    long = long.merge(values, on=["style_id", "feature"])
    long["abs"] = long.contribution.abs()
    long = long.sort_values(["style_id", "abs"], ascending=[True, False]).groupby("style_id").head(TOP_CONTRIBUTIONS)
    return pred, long.drop(columns="abs")


def export_api_data(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Everything the backend API needs besides the predictions themselves:

    - style_history:   success score, revenue and units per style per past quarter
    - style_articles:  each style's best-selling articles last quarter
    - season_winners:  the top 3 styles of every season, actual for past quarters (among commercially relevant styles,
                       the same business rule as the model) and predicted for the prediction quarter
    - season_articles: the 5 best-selling articles of each of those styles *in that season*
    """
    styled = add_style_columns(load_articles())
    tx = load_transactions(columns=["t_dat", "article_id", "price"])
    tx = tx[tx.article_id.isin(styled.article_id) & (tx.t_dat <= LAST_TRAINING_DAY)]
    quarters = quarters_between(tx.t_dat.min(), LAST_TRAINING_DAY)
    full = build_style_quarter_table(tx, styled, quarters).sort_values(["style_id", "start"])

    history = full[full.style_id.isin(pred.style_id)][
        ["style_id", "quarter_id", "season", "start", "success_score", "revenue", "units", "articles", "full_price_factor"]]

    last_quarter_units = tx[tx.t_dat >= quarters.start.iloc[-1]].groupby("article_id").size()
    total_units = tx.groupby("article_id").size()
    articles = styled[styled.style_id.isin(pred.style_id)][["style_id", "article_id", "prod_name", "detail_desc"]].copy()
    articles["units_last_quarter"] = articles.article_id.map(last_quarter_units).fillna(0).astype(int)
    articles["units_total"] = articles.article_id.map(total_units).fillna(0).astype(int)
    articles = (articles.sort_values(["style_id", "units_last_quarter", "units_total"], ascending=[True, False, False])
                .groupby("style_id").head(ARTICLES_PER_STYLE))

    # past seasons: the actual top 3 among commercially relevant styles (revenue last quarter in the top 10%);
    # the first quarter has no previous quarter, so the rule can't be applied and it is left out
    full["revenue_prev"] = full.groupby("style_id").revenue.shift(1)
    cutoff = full[full.revenue_prev > 0].groupby("quarter_id").revenue_prev.quantile(1 - COMMERCIAL_TOP_SHARE)
    relevant = full[full.revenue_prev >= full.quarter_id.map(cutoff)]
    past = relevant.sort_values("success_score", ascending=False).groupby("quarter_id").head(TOP_STYLES).copy()
    past["rank"] = past.groupby("quarter_id").success_score.rank(ascending=False, method="first").astype(int)
    past["status"] = "actual"

    nxt = predict_quarter_after(LAST_TRAINING_DAY)
    predicted = pred[pred["rank"].notna() & (pred["rank"] <= TOP_STYLES)].copy()
    predicted = predicted.assign(season=nxt["season"], start=nxt["start"], end=nxt["end"], status="predicted",
                                 success_score=predicted.prediction_score, rank=predicted["rank"].astype(int),
                                 revenue=np.nan, units=np.nan)
    cols = ["quarter_id", "season", "start", "end", "status", "rank", "style_id", "style_name", "group_name",
            "success_score", "revenue", "units"]
    winners = pd.concat([past[cols], predicted[cols]]).sort_values(["start", "rank"], ignore_index=True)

    # each winner's best-selling articles in its season (for the predicted season: in the last quarter)
    rows = []
    for q in quarters.itertuples():
        styles_q = past.loc[past.quarter_id == q.quarter_id, "style_id"]
        if styles_q.empty:
            continue
        units_q = tx[(tx.t_dat >= q.start) & (tx.t_dat <= q.end)].groupby("article_id").size().rename("units")
        a = styled[styled.style_id.isin(styles_q)][["style_id", "article_id", "prod_name", "detail_desc"]].join(units_q, on="article_id")
        rows.append(a.dropna(subset=["units"]).sort_values("units", ascending=False)
                    .groupby("style_id").head(ARTICLES_PER_STYLE).assign(quarter_id=q.quarter_id))
    rows.append(articles[articles.style_id.isin(predicted.style_id)]
                .rename(columns={"units_last_quarter": "units"}).drop(columns="units_total")
                .assign(quarter_id=nxt["quarter_id"]))
    season_articles = pd.concat(rows, ignore_index=True)
    season_articles["units"] = season_articles.units.astype(int)
    season_articles = season_articles.merge(winners[["quarter_id", "style_id", "rank"]], on=["quarter_id", "style_id"])

    return {"style_history": history, "style_articles": articles,
            "season_winners": winners, "season_articles": season_articles}


def images_needed(season_articles: pd.DataFrame) -> pd.DataFrame:
    """The article images the app shows: the best-selling articles of every season's top styles."""
    need = season_articles[["style_id", "article_id"]].drop_duplicates().copy()
    need["article_id"] = need.article_id.map(lambda a: f"{int(a):010d}")
    need["kaggle_path"] = "images/" + need.article_id.str[:3] + "/" + need.article_id + ".jpg"
    return need.sort_values(["style_id", "article_id"], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-images", action="store_true", help="don't fetch the images of the seasons' top styles")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_PATH)
    pred, explanations = predict_quarter(bundle, load_or_build_features())
    OUTPUTS.mkdir(exist_ok=True)
    pred.to_csv(OUTPUTS / "predictions.csv", index=False)
    explanations.to_parquet(OUTPUTS / "explanations.parquet", index=False)
    exported = export_api_data(pred)
    for name, table in exported.items():
        table.to_parquet(OUTPUTS / f"{name}.parquet", index=False)
    images_needed(exported["season_articles"]).to_csv(OUTPUTS / "images_needed.csv", index=False)

    print(f"{pred.quarter_id.iloc[0]}: scored {len(pred):,} styles (model trained on {', '.join(bundle['trained_on'])})")
    print(f"commercially relevant styles: {int(pred.commercially_relevant.sum()):,}")
    print(pred[pred["rank"].notna()].head(10).round(3).to_string(index=False))

    if not args.no_images:
        try:
            fetch()
        except ImageSourceUnavailable as error:   # predictions are done; images can be fetched later
            print(f"WARNING: images not fetched. {error}")


if __name__ == "__main__":
    main()
