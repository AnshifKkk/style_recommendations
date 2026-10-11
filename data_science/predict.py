"""Score every style for the prediction quarter and explain each prediction.

    python -m data_science.predict

Writes (these files are everything the backend API needs; it never reads raw data or training code)
- outputs/predictions.csv          one row per style: predicted success score, rank among commercially relevant styles
- outputs/explanations.parquet     the largest feature contributions per style
- outputs/style_history.parquet    success score, revenue, units per style per past quarter
- outputs/style_articles.parquet   each style's best-selling articles (name, description, units)

Explanations: the model is linear in log(1 + success score), so each prediction is exactly
    average prediction + Σ contribution(feature),   contribution = coefficient × (value − training average)
summed over the encoded columns of a feature. For a linear model this equals its SHAP values.
"""
import joblib
import numpy as np
import pandas as pd

from data_science.data import load_articles, load_transactions
from data_science.features.build import CATEGORICAL_FEATURES, FEATURES, TARGET
from data_science.features.quarters import build_style_quarter_table, quarters_between
from data_science.features.style import STYLE_ATTRIBUTES, add_style_columns
from data_science.train import (LAST_TRAINING_DAY, MODEL_PATH, OUTPUTS, as_model_input, commercially_relevant,
                                load_or_build_features)

TOP_CONTRIBUTIONS = 8
ARTICLES_PER_STYLE = 5


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


def export_api_data(style_ids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarterly history and best-selling articles of the given styles, for the backend API."""
    styled = add_style_columns(load_articles())
    tx = load_transactions(columns=["t_dat", "article_id", "price"])
    tx = tx[tx.article_id.isin(styled.article_id) & (tx.t_dat <= LAST_TRAINING_DAY)]
    quarters = quarters_between(tx.t_dat.min(), LAST_TRAINING_DAY)

    history = build_style_quarter_table(tx, styled, quarters)
    history = history[history.style_id.isin(style_ids)][
        ["style_id", "quarter_id", "season", "start", "success_score", "revenue", "units", "articles", "full_price_factor"]]

    last_quarter_units = tx[tx.t_dat >= quarters.start.iloc[-1]].groupby("article_id").size()
    total_units = tx.groupby("article_id").size()
    articles = styled[styled.style_id.isin(style_ids)][["style_id", "article_id", "prod_name", "detail_desc"]].copy()
    articles["units_last_quarter"] = articles.article_id.map(last_quarter_units).fillna(0).astype(int)
    articles["units_total"] = articles.article_id.map(total_units).fillna(0).astype(int)
    articles = (articles.sort_values(["style_id", "units_last_quarter", "units_total"], ascending=[True, False, False])
                .groupby("style_id").head(ARTICLES_PER_STYLE))
    return history, articles


def main():
    bundle = joblib.load(MODEL_PATH)
    pred, explanations = predict_quarter(bundle, load_or_build_features())
    OUTPUTS.mkdir(exist_ok=True)
    pred.to_csv(OUTPUTS / "predictions.csv", index=False)
    explanations.to_parquet(OUTPUTS / "explanations.parquet", index=False)
    history, articles = export_api_data(pred.style_id)
    history.to_parquet(OUTPUTS / "style_history.parquet", index=False)
    articles.to_parquet(OUTPUTS / "style_articles.parquet", index=False)

    print(f"{pred.quarter_id.iloc[0]}: scored {len(pred):,} styles (model trained on {', '.join(bundle['trained_on'])})")
    print(f"commercially relevant styles: {int(pred.commercially_relevant.sum()):,}")
    print(pred[pred["rank"].notna()].head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
