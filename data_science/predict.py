"""Score every style for the prediction quarter and explain each prediction.

    python -m data_science.predict

Writes
- outputs/predictions.csv     one row per style: predicted success score and rank
- outputs/explanations.csv    the largest feature contributions per style

Explanations: the model is linear in log(1 + success score), so each prediction is exactly
    average prediction + Σ contribution(feature),   contribution = coefficient × (value − training average)
summed over the encoded columns of a feature. For a linear model this equals its SHAP values.
"""
import joblib
import numpy as np
import pandas as pd

from data_science.features.build import CATEGORICAL_FEATURES, FEATURES, TARGET
from data_science.train import MODEL_PATH, OUTPUTS, as_model_input, load_or_build_features

TOP_CONTRIBUTIONS = 8


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


def main():
    bundle = joblib.load(MODEL_PATH)
    features = load_or_build_features()
    rows = features[features[TARGET].isna()].copy()           # the prediction quarter
    quarter_id = rows.quarter_id.iloc[0]

    log_pred, contrib = explain(bundle, rows)
    rows["prediction_score"] = np.expm1(log_pred)
    rows["rank"] = rows.prediction_score.rank(ascending=False, method="first").astype(int)
    pred = rows.sort_values("rank")[["rank", "style_id", "style_name", "group_name", "quarter_id",
                                     "prediction_score", "score_lag1", "score_lag4"]]
    OUTPUTS.mkdir(exist_ok=True)
    pred.to_csv(OUTPUTS / "predictions.csv", index=False)

    long = contrib.assign(style_id=rows.style_id).melt(id_vars="style_id", var_name="feature", value_name="contribution")
    values = rows[["style_id", *FEATURES]].astype({c: str for c in FEATURES}).melt(
        id_vars="style_id", var_name="feature", value_name="feature_value")
    long = long.merge(values, on=["style_id", "feature"])
    long["abs"] = long.contribution.abs()
    long = long.sort_values(["style_id", "abs"], ascending=[True, False]).groupby("style_id").head(TOP_CONTRIBUTIONS)
    long.drop(columns="abs").to_csv(OUTPUTS / "explanations.csv", index=False)

    print(f"{quarter_id}: scored {len(pred):,} styles (model trained on {', '.join(bundle['trained_on'])})")
    print(pred.head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
