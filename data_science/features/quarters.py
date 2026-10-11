"""Fashion quarters and the style × quarter table (one row per style per quarter, with the success target).

Quarters follow the fashion seasons: Spring (Mar–May), Summer (Jun–Aug), Autumn (Sep–Nov), Winter (Dec–Feb).
A winter quarter is named after the year it starts in (winter 2019 = Dec 2019 – Feb 2020).
"""
import pandas as pd

from data_science.features.style import STYLE_ATTRIBUTES
from data_science.features.success import success_scores

SEASON_START_MONTH = {"spring": 3, "summer": 6, "autumn": 9, "winter": 12}


def quarter(year: int, season: str) -> dict:
    start = pd.Timestamp(year=year, month=SEASON_START_MONTH[season], day=1)
    end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    return {"quarter_id": f"{year}_{season}", "season": season, "start": start, "end": end}


def quarters_between(first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """All quarters overlapping [first, last], oldest first."""
    rows = []
    for year in range(first.year - 1, last.year + 1):
        for season in SEASON_START_MONTH:
            q = quarter(year, season)
            if q["end"] >= first and q["start"] <= last:
                rows.append(q)
    return pd.DataFrame(rows).sort_values("start", ignore_index=True)


def build_style_quarter_table(tx: pd.DataFrame, styled_articles: pd.DataFrame,
                              quarters: pd.DataFrame) -> pd.DataFrame:
    """One row per (style, quarter) for every style in the catalogue and every given quarter.

    Target columns come from `success_scores` over the part of the quarter covered by the data.
    A style with no sales in a quarter gets success_score = 0 and revenue/units/articles = 0.
    """
    quarters = quarters.copy()
    quarters["days_observed"] = [(min(e, tx.t_dat.max()) - max(s, tx.t_dat.min())).days + 1
                                 for s, e in zip(quarters.start, quarters.end)]
    per_quarter = []
    for q in quarters.itertuples():
        s = success_scores(tx, styled_articles, max(q.start, tx.t_dat.min()), min(q.end, tx.t_dat.max()))
        s["quarter_id"] = q.quarter_id
        per_quarter.append(s)
    scores = pd.concat(per_quarter, ignore_index=True)

    styles = (styled_articles.drop_duplicates("style_id")
              [["style_id", "style_name", "group_id", "group_name", *STYLE_ATTRIBUTES]])
    grid = styles.merge(quarters[["quarter_id", "season", "start", "end", "days_observed"]], how="cross")
    target_cols = ["success_score", "revenue", "full_price_factor", "units", "articles"]
    table = grid.merge(scores[["style_id", "quarter_id", *target_cols]], on=["style_id", "quarter_id"], how="left")
    table["sold"] = table.revenue.notna()
    table[["success_score", "revenue", "units", "articles"]] = table[["success_score", "revenue", "units", "articles"]].fillna(0)
    return table.sort_values(["style_id", "start"], ignore_index=True)
