"""Model evaluation (walk-forward) and training of the final model.

For each evaluation quarter Q, models are trained only on quarters before Q and predict every style that has
sold before Q. Features only use data from before each row's own quarter (see features/build.py).

    python -m data_science.train                      # evaluate, then train the final model on all quarters
    python -m data_science.train --rebuild-features   # rebuild the feature table first
"""
import argparse
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_science.data import CACHE_DIR, ROOT, load_articles, load_transactions
from data_science.features.build import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET,
                                         build_feature_table, predict_quarter_after)
from data_science.features.quarters import quarters_between
from data_science.features.style import add_style_columns

LAST_TRAINING_DAY = pd.Timestamp("2020-08-31")
FEATURES_PATH = CACHE_DIR / "features.parquet"
OUTPUTS = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "ridge.joblib"

VALIDATION_QUARTERS = ["2019_winter", "2020_spring"]   # used to choose the model
TEST_QUARTERS = ["2020_summer"]                         # final check, not used for any choice

LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=42, verbose=-1)


# ---------------------------------------------------------------- data
def load_or_build_features(rebuild: bool = False) -> pd.DataFrame:
    if FEATURES_PATH.exists() and not rebuild:
        f = pd.read_parquet(FEATURES_PATH)
        for col in CATEGORICAL_FEATURES:
            f[col] = f[col].astype("category")
        return f
    styled = add_style_columns(load_articles())
    tx = load_transactions()
    tx = tx[tx.article_id.isin(styled.article_id) & (tx.t_dat <= LAST_TRAINING_DAY)].reset_index(drop=True)
    quarters = quarters_between(tx.t_dat.min(), LAST_TRAINING_DAY)
    f = build_feature_table(tx, styled, quarters, predict_quarter_after(LAST_TRAINING_DAY))
    f.to_parquet(FEATURES_PATH, index=False)
    return f


# ---------------------------------------------------------------- metric
def top3_quality(y_true, y_pred) -> float:
    """Average actual success score of the 3 styles we pick ÷ average actual score of the true top 3.

    1.0 means our three picks did as well as the best three styles that quarter.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    picked = np.argsort(-y_pred, kind="stable")[:3]
    best = np.sort(y_true)[::-1][:3]
    return y_true[picked].mean() / best.mean()


# ---------------------------------------------------------------- models
def baseline_last_quarter(train, test):
    return test.score_lag1


def make_ridge() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))])


def as_model_input(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].astype({c: str for c in CATEGORICAL_FEATURES})


def ridge(train, test):
    model = make_ridge().fit(as_model_input(train), np.log1p(train[TARGET]))
    return model.predict(as_model_input(test))


def lgbm_regression(train, test):
    model = lgb.LGBMRegressor(objective="regression", **LGBM_PARAMS)
    model.fit(train[FEATURES], np.log1p(train[TARGET]), categorical_feature=CATEGORICAL_FEATURES)
    return model.predict(test[FEATURES])


MODELS = {"baseline: last quarter": baseline_last_quarter, "ridge": ridge, "lightgbm": lgbm_regression}


# ---------------------------------------------------------------- walk-forward
def walk_forward(features: pd.DataFrame, quarters: list[str]) -> pd.DataFrame:
    labelled = features[features[TARGET].notna()]
    rows = []
    for qid in quarters:
        qi = labelled.loc[labelled.quarter_id == qid, "qi"].iloc[0]
        train, test = labelled[labelled.qi < qi], labelled[labelled.qi == qi]
        for name, fn in MODELS.items():
            rows.append({"quarter": qid, "model": name,
                         "top3_quality": top3_quality(test[TARGET], fn(train, test))})
    return pd.DataFrame(rows)


def train_final_model(features: pd.DataFrame) -> Pipeline:
    """Ridge trained on every labelled quarter; saved with what predict.py needs for explanations."""
    labelled = features[features[TARGET].notna()]
    model = make_ridge().fit(as_model_input(labelled), np.log1p(labelled[TARGET]))
    train_matrix = model.named_steps["pre"].transform(as_model_input(labelled))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "train_mean": np.asarray(train_matrix.mean(axis=0)).ravel(),
                 "trained_on": sorted(labelled.quarter_id.unique())}, MODEL_PATH)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    features = load_or_build_features(args.rebuild_features)
    results = walk_forward(features, VALIDATION_QUARTERS + TEST_QUARTERS)
    results["split"] = np.where(results.quarter.isin(TEST_QUARTERS), "test", "validation")
    OUTPUTS.mkdir(exist_ok=True)
    results.to_csv(OUTPUTS / "evaluation.csv", index=False)
    table = results.pivot(index="model", columns="quarter", values="top3_quality").loc[list(MODELS)]
    table["validation mean"] = table[VALIDATION_QUARTERS].mean(axis=1)
    print("top-3 quality (1.0 = our three picks did as well as the best three styles)")
    print(table[VALIDATION_QUARTERS + ["validation mean"] + TEST_QUARTERS].round(3).to_string())

    train_final_model(features)
    print(f"\nfinal model trained on all labelled quarters → {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
