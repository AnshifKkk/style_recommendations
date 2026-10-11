"""Success metric for a style in a season window.

    Adjusted revenue = Revenue × (Revenue ÷ Full-price revenue)
    Group benchmark  = Σ Adjusted revenue² ÷ Σ Adjusted revenue    (over styles in the same group)
    Success score    = Adjusted revenue ÷ Group benchmark

Full-price revenue values every unit at its article's highest observed price (up to the end of the
window), so discounting lowers the score. The group benchmark is the revenue-weighted typical style
size in the category, so a long tail of tiny styles barely moves it. A score of 1 means "as good as
the typical style a sale in this category comes from".
"""
import pandas as pd


def success_scores(tx: pd.DataFrame, styled_articles: pd.DataFrame,
                   start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """One row per style that sold in [start, end] (inclusive), sorted by success score.

    `styled_articles` is the articles table after `features.style.add_style_columns`.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    # reference (full) price per article: highest price observed up to the end of the window
    ref_price = tx.loc[tx.t_dat <= end].groupby("article_id")["price"].max().rename("ref_price")

    window = tx.loc[(tx.t_dat >= start) & (tx.t_dat <= end), ["article_id", "price"]]
    window = window.join(ref_price, on="article_id").merge(
        styled_articles[["article_id", "style_id", "style_name", "group_id", "group_name"]],
        on="article_id")

    s = window.groupby(["style_id", "style_name", "group_id", "group_name"], as_index=False).agg(
        revenue=("price", "sum"), full_price_revenue=("ref_price", "sum"),
        units=("price", "size"), articles=("article_id", "nunique"))

    s["full_price_factor"] = s.revenue / s.full_price_revenue
    s["adjusted_revenue"] = s.revenue * s.full_price_factor

    grp = s.groupby("group_id")["adjusted_revenue"]
    s["group_styles"] = grp.transform("size")
    s["group_benchmark"] = grp.transform(lambda a: (a ** 2).sum() / a.sum())
    s["success_score"] = s.adjusted_revenue / s.group_benchmark

    return s.sort_values("success_score", ascending=False, ignore_index=True)
