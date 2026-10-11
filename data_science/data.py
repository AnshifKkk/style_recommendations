"""Loading the raw Kaggle tables.

The raw CSVs live in `kaggle_downlods/` (git-ignored). The 3.5 GB transactions CSV is converted
once to a Parquet cache in `data/`, so later loads take seconds and can read only the needed columns.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from data_science.features.style import STYLE_ATTRIBUTES

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "kaggle_downlods"
CACHE_DIR = ROOT / "data"

TX_COLUMNS = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]

# Product groups that are not fashion and have no "style" to predict (~0.01% of revenue in total).
NON_FASHION_GROUPS = ["Items", "Cosmetic", "Furniture", "Garment and Shoe care",
                      "Interior textile", "Stationery", "Fun"]


# Attribute values that mean "not recorded"; an article with one of these in a style attribute has no defined style.
UNKNOWN_VALUES = ["unknown", "undefined"]


def load_articles(raw_dir: Path = RAW_DIR, clean: bool = True) -> pd.DataFrame:
    """Article attributes, with `article_id` as int64 (matching transactions).

    With `clean=True` (default) two groups of articles are removed (~0.4% of revenue together):
    - non-fashion product groups (`NON_FASHION_GROUPS`),
    - articles with an unknown/undefined value in any style attribute.
    Transactions of removed articles drop out wherever transactions are joined to articles.
    """
    art = pd.read_csv(raw_dir / "articles.csv", dtype={"article_id": str, "product_code": str})
    art["article_id"] = art["article_id"].astype("int64")
    if clean:
        non_fashion = art.product_group_name.isin(NON_FASHION_GROUPS)
        unknown = art[STYLE_ATTRIBUTES].apply(lambda c: c.astype(str).str.lower().isin(UNKNOWN_VALUES)).any(axis=1)
        art = art[~non_fashion & ~unknown].reset_index(drop=True)
    return art


def _build_transactions_cache(raw_dir: Path, cache: Path) -> None:
    types = {"t_dat": pa.date32(), "customer_id": pa.string(), "article_id": pa.int64(),
             "price": pa.float32(), "sales_channel_id": pa.int8()}
    table = pacsv.read_csv(raw_dir / "transactions_train.csv",
                           convert_options=pacsv.ConvertOptions(column_types=types))
    cache.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, cache)


def load_transactions(columns: list[str] | None = None,
                      raw_dir: Path = RAW_DIR, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Transactions, one row per unit sold. Pass `columns` to load only what you need
    (customer_id alone is ~2 GB in memory)."""
    cache = cache_dir / "transactions.parquet"
    if not cache.exists() or "customer_id" not in pq.read_schema(cache).names:
        _build_transactions_cache(raw_dir, cache)
    tx = pd.read_parquet(cache, columns=columns or TX_COLUMNS)
    if "t_dat" in tx:
        tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    return tx
