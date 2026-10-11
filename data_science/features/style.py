"""Style definition.

A style is the combination of attributes that together define a customer's style choice:
who it is for, what it is, its surface design, its colour and its colour tone.
"""
import re

import pandas as pd

STYLE_ATTRIBUTES = [
    "index_group_name",             # who it is for: Ladieswear / Menswear / Divided / Baby-Children / Sport
    "product_type_name",            # what it is: Dress, Trousers, Hoodie, ...
    "graphical_appearance_name",    # surface design: Solid, Stripe, Denim, All over pattern, ...
    "colour_group_name",            # specific colour: Black, Light Pink, Dark Blue, ...
    "perceived_colour_value_name",  # tone: Dark, Light, Dusty Light, Bright, ...
]

# Peer group used to normalise for category demand.
GROUP_ATTRIBUTES = ["index_group_name", "product_type_name"]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def make_id(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Stable, URL-safe id, e.g. `ladieswear__dress__all-over-pattern__light-pink__dusty-light`."""
    return df[columns].astype(str).apply(lambda row: "__".join(_slug(v) for v in row), axis=1)


def make_name(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Human-readable name, e.g. `Ladieswear · Dress · All over pattern · Light Pink · Dusty Light`."""
    return df[columns].astype(str).agg(" · ".join, axis=1)


def _unify_case_variants(values: pd.Series) -> pd.Series:
    """Maps spellings that differ only in case (e.g. 'Dog Wear' / 'Dog wear') to the most common one."""
    canonical = (values.groupby(values.str.lower()).agg(lambda v: v.value_counts().index[0]))
    return values.str.lower().map(canonical)


def add_style_columns(articles: pd.DataFrame) -> pd.DataFrame:
    """Adds style_id, style_name, group_id and group_name to the articles table."""
    out = articles.copy()
    for col in STYLE_ATTRIBUTES:
        out[col] = _unify_case_variants(out[col].astype(str))
    out["style_id"] = make_id(out, STYLE_ATTRIBUTES)
    out["style_name"] = make_name(out, STYLE_ATTRIBUTES)
    out["group_id"] = make_id(out, GROUP_ATTRIBUTES)
    out["group_name"] = make_name(out, GROUP_ATTRIBUTES)
    return out
