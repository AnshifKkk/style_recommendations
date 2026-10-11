"""Feature engineering: one row per (style, target quarter), using only data from before the quarter starts.

Feature groups
- A. style history:        success score lags, same quarter last year, mean/max/trend, age, size, discounting
- B. recent momentum:      last 4 weeks before the quarter, new articles launched in the last 13 weeks
- C. seasonality:          target season, share of past revenue earned in that season (style and category)
- D. attributes:           the five style attributes (categorical)
- E. attribute momentum:   is the style's colour / pattern / tone gaining share across all categories?
- F. category context:     category growth, number of competing styles, category benchmark
- G. customers and price:  buyers, repeat-purchase rate, online share, price position within the category
"""
import numpy as np
import pandas as pd

from data_science.features.quarters import build_style_quarter_table, quarter
from data_science.features.style import STYLE_ATTRIBUTES
from data_science.features.success import success_scores

SEASONS = ["spring", "summer", "autumn", "winter"]
CATEGORICAL_FEATURES = STYLE_ATTRIBUTES + ["target_season"]
NUMERIC_FEATURES = [
    # A
    "score_lag1", "score_lag2", "score_lag3", "score_lag4", "score_mean4", "score_max4", "score_trend4",
    "quarters_sold4", "age_quarters", "quarters_since_last_sale",
    "log_revenue_lag1", "log_units_lag1", "articles_lag1", "full_price_factor_lag1",
    # B
    "score_last4w", "revenue_momentum", "new_articles_13w",
    # C
    "style_season_share", "category_season_share",
    # E
    "colour_momentum", "pattern_momentum", "tone_momentum",
    # F
    "category_growth", "category_n_styles_lag1", "category_benchmark_lag1",
    # G
    "log_customers_lag1", "repeat_rate4", "online_share_lag1", "price_index_lag1",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "success_score"


def _add_history_features(t: pd.DataFrame) -> pd.DataFrame:
    """Groups A, C, E, F from the style × quarter table (rows sorted by style, then quarter)."""
    g = t.groupby("style_id", sort=False)
    sold = t.sold.astype(float)

    for k in range(1, 5):
        t[f"score_lag{k}"] = g.success_score.shift(k)
    lags = t[[f"score_lag{k}" for k in range(1, 5)]]
    t["score_mean4"] = lags.mean(axis=1)
    t["score_max4"] = lags.max(axis=1)
    t["score_trend4"] = lags[["score_lag1", "score_lag2"]].mean(axis=1) - lags[["score_lag3", "score_lag4"]].mean(axis=1)
    t["quarters_sold4"] = sum(sold.groupby(t.style_id).shift(k) for k in range(1, 5))

    first_sold_qi = t.qi.where(t.sold).groupby(t.style_id).transform("min")
    t["age_quarters"] = (t.qi - first_sold_qi).where(first_sold_qi < t.qi)
    last_sold_before = t.qi.where(t.sold).groupby(t.style_id).shift(1).groupby(t.style_id).ffill()
    t["quarters_since_last_sale"] = t.qi - last_sold_before

    t["log_revenue_lag1"] = np.log1p(g.revenue.shift(1))
    t["log_units_lag1"] = np.log1p(g.units.shift(1))
    t["articles_lag1"] = g.articles.shift(1)
    t["full_price_factor_lag1"] = g.full_price_factor.shift(1)

    # C. seasonality: share of all past revenue that was earned in the target quarter's season
    rev = t.revenue.fillna(0.0)   # the prediction quarter has no revenue yet
    past_total = rev.groupby(t.style_id).cumsum() - rev
    past_in_season = pd.Series(0.0, index=t.index)
    for s in SEASONS:
        rev_s = rev.where(t.season == s, 0.0)
        cum_s = rev_s.groupby(t.style_id).cumsum() - rev_s
        past_in_season = past_in_season.where(t.season != s, cum_s)
    t["style_season_share"] = (past_in_season / past_total).where(past_total > 0)

    # category (group) level per quarter
    t["_adj"] = t.revenue * t.full_price_factor.fillna(0)
    cat = t.groupby(["group_id", "qi"]).agg(revenue=("revenue", "sum"), n_styles=("sold", "sum"),
                                           sum_a=("_adj", "sum"), sum_a2=("_adj", lambda a: (a ** 2).sum()),
                                           season=("season", "first")).reset_index()
    cat["benchmark"] = cat.sum_a2 / cat.sum_a.replace(0, np.nan)
    cat = cat.sort_values(["group_id", "qi"])
    cg = cat.groupby("group_id")
    cat["category_n_styles_lag1"] = cg.n_styles.shift(1)
    cat["category_benchmark_lag1"] = cg.benchmark.shift(1)
    cat["category_growth"] = cg.revenue.shift(1) / cg.revenue.shift(5)
    past_total = cg.revenue.cumsum() - cat.revenue
    past_in_season = pd.Series(0.0, index=cat.index)
    for s in SEASONS:
        rev_s = cat.revenue.where(cat.season == s, 0.0)
        cum_s = rev_s.groupby(cat.group_id).cumsum() - rev_s
        past_in_season = past_in_season.where(cat.season != s, cum_s)
    cat["category_season_share"] = (past_in_season / past_total).where(past_total > 0)
    t = t.merge(cat[["group_id", "qi", "category_n_styles_lag1", "category_benchmark_lag1",
                     "category_growth", "category_season_share"]], on=["group_id", "qi"], how="left")

    # E. attribute momentum: share of all revenue last quarter ÷ the same quarter a year before that
    total = t.groupby("qi").revenue.sum()
    for attr, name in [("colour_group_name", "colour_momentum"), ("graphical_appearance_name", "pattern_momentum"),
                       ("perceived_colour_value_name", "tone_momentum")]:
        share = (t.groupby([attr, "qi"]).revenue.sum() / total).rename("share").reset_index()
        share = share.pivot(index=attr, columns="qi", values="share")
        mom = (share.shift(1, axis=1) / share.shift(5, axis=1)).stack().rename(name).reset_index()
        t = t.merge(mom, on=[attr, "qi"], how="left")

    return t.drop(columns="_adj")


def _add_transaction_features(t: pd.DataFrame, tx: pd.DataFrame, styled: pd.DataFrame,
                              quarters: pd.DataFrame) -> pd.DataFrame:
    """Groups B and G, computed from raw transactions in windows just before each quarter starts."""
    codes = pd.Series(np.arange(styled.style_id.nunique()), index=styled.style_id.unique())
    art_style = styled.set_index("article_id").style_id.map(codes)
    art_group = styled.set_index("article_id").group_id
    style_group = styled.drop_duplicates("style_id").set_index("style_id").group_id

    x = pd.DataFrame({"t_dat": tx.t_dat.values, "article_id": tx.article_id.values, "price": tx.price.values,
                      "online": (tx.sales_channel_id == 2).values,
                      "customer": pd.factorize(tx.customer_id)[0].astype("int32")})
    x["style"] = x.article_id.map(art_style).astype("int32")
    first_sale = x.groupby("article_id").t_dat.min()
    art_first = pd.DataFrame({"first_sale": first_sale, "style": art_style.reindex(first_sale.index)})

    rows = []
    for q in quarters.itertuples():
        c = q.start
        last4w = success_scores(tx, styled, c - pd.Timedelta(days=28), c - pd.Timedelta(days=1))
        f = pd.DataFrame(index=codes.values)
        # no sales in the window means a score / revenue / buyer count of 0, not "unknown"
        f["score_last4w"] = last4w.set_index(last4w.style_id.map(codes)).success_score.reindex(f.index).fillna(0)
        rev4w = last4w.set_index(last4w.style_id.map(codes)).revenue.reindex(f.index).fillna(0)

        w13 = x[(x.t_dat >= c - pd.Timedelta(days=91)) & (x.t_dat < c)]
        by = w13.groupby("style")
        rev13 = by.price.sum()
        f["revenue_momentum"] = (rev4w / 4) / (rev13 / 13)
        f["log_customers_lag1"] = np.log1p(by.customer.nunique().reindex(f.index).fillna(0))
        f["online_share_lag1"] = by.online.mean()
        avg_price = rev13 / by.size()
        grp = pd.Series(style_group.reindex(codes.index).values, index=codes.values)
        grp_price = w13.price.groupby(grp.reindex(w13["style"]).values).mean()
        f["price_index_lag1"] = avg_price / grp.reindex(avg_price.index).map(grp_price).values

        new = art_first[(art_first.first_sale >= c - pd.Timedelta(days=91)) & (art_first.first_sale < c)]
        f["new_articles_13w"] = new.groupby("style").size()
        f["new_articles_13w"] = f.new_articles_13w.fillna(0)

        w52 = x[(x.t_dat >= c - pd.Timedelta(days=364)) & (x.t_dat < c)]
        days_per_buyer = w52.groupby(["style", "customer"]).t_dat.nunique()
        f["repeat_rate4"] = (days_per_buyer > 1).groupby(level="style").mean()

        f["qi"] = q.qi
        f["style_id"] = codes.index
        rows.append(f)

    feats = pd.concat(rows, ignore_index=True)
    return t.merge(feats, on=["style_id", "qi"], how="left")


def build_feature_table(tx: pd.DataFrame, styled: pd.DataFrame, quarters: pd.DataFrame,
                        predict_quarter: dict) -> pd.DataFrame:
    """Feature table for every (style, quarter) from the second quarter on, plus `predict_quarter`
    (which has no target). Only rows where the style sold at least once before the quarter are kept.

    `tx` must only contain data from before `predict_quarter` starts and needs the columns
    t_dat, article_id, price, sales_channel_id and customer_id.
    """
    table = build_style_quarter_table(tx[["t_dat", "article_id", "price"]], styled, quarters)
    future = table[table.quarter_id == quarters.quarter_id.iloc[-1]].copy()
    future[["quarter_id", "season", "start", "end"]] = [predict_quarter["quarter_id"], predict_quarter["season"],
                                                        predict_quarter["start"], predict_quarter["end"]]
    future[["success_score", "revenue", "full_price_factor", "units", "articles", "days_observed"]] = np.nan
    future["sold"] = False
    table = pd.concat([table, future], ignore_index=True)

    order = pd.concat([quarters[["quarter_id", "start"]],
                       pd.DataFrame([{"quarter_id": predict_quarter["quarter_id"], "start": predict_quarter["start"]}])])
    qi = {qid: i for i, qid in enumerate(order.sort_values("start").quarter_id)}
    table["qi"] = table.quarter_id.map(qi)
    table = table.sort_values(["style_id", "qi"], ignore_index=True)

    table = _add_history_features(table)
    target_quarters = order[order.quarter_id.map(qi) >= 1].assign(qi=lambda d: d.quarter_id.map(qi))
    table = _add_transaction_features(table, tx, styled, target_quarters)

    table[NUMERIC_FEATURES] = table[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)   # ratios with a zero base
    table["target_season"] = table.season
    table = table[(table.qi >= 1) & table.age_quarters.notna()].reset_index(drop=True)
    for col in CATEGORICAL_FEATURES:
        table[col] = table[col].astype("category")
    return table


def predict_quarter_after(last_day: pd.Timestamp) -> dict:
    """The quarter starting the day after `last_day`."""
    nxt = last_day + pd.Timedelta(days=1)
    season = {3: "spring", 6: "summer", 9: "autumn", 12: "winter"}[nxt.month]
    return quarter(nxt.year, season)
