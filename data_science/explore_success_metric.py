import sys, os, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 70)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "kaggle_downlods")
CACHE = os.path.join(ROOT, "data", "transactions.parquet")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)

STYLE_COLS = ["index_group_name", "product_type_name", "graphical_appearance_name",
              "colour_group_name", "perceived_colour_value_name"]

# ---------- load ----------
art = pd.read_csv(os.path.join(DATA, "articles.csv"), dtype={"article_id": str},
                  usecols=["article_id"] + STYLE_COLS)
art["article_id"] = art["article_id"].astype(np.int64)
art["style_id"] = art[STYLE_COLS].astype(str).agg(" · ".join, axis=1)
art["group"] = art["index_group_name"] + " · " + art["product_type_name"]

t0 = time.time()
if os.path.exists(CACHE):
    tx = pd.read_parquet(CACHE)
else:
    tx = pd.read_csv(os.path.join(DATA, "transactions_train.csv"),
                     usecols=["t_dat", "article_id", "price"], engine="pyarrow",
                     dtype={"article_id": "int64", "price": "float32"})
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    tx.to_parquet(CACHE)
print(f"transactions: {len(tx):,} rows, {tx.t_dat.min().date()} → {tx.t_dat.max().date()}  ({time.time()-t0:.0f}s)")

# ---------- price sanity ----------
pstat = tx.groupby("article_id")["price"].agg(["max", "median"])
r = pstat["max"] / pstat["median"]
print(f"max/median price per article: p50={r.quantile(.5):.2f} p90={r.quantile(.9):.2f} p99={r.quantile(.99):.2f}")


def season(start, end, label):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hist = tx[tx.t_dat <= end]
    ref = hist.groupby("article_id")["price"].max().rename("ref")          # P_a
    w = tx[(tx.t_dat >= start) & (tx.t_dat <= end)].join(ref, on="article_id")
    w = w.merge(art[["article_id", "style_id", "group"]], on="article_id")
    s = w.groupby(["style_id", "group"]).agg(
        revenue=("price", "sum"), full_rev=("ref", "sum"),
        units=("price", "size"), articles=("article_id", "nunique")).reset_index()
    s["F"] = s.revenue / s.full_rev
    s["A"] = s.revenue * s.F
    g = s.groupby("group").agg(sumA=("A", "sum"), sumA2=("A", lambda x: (x**2).sum()),
                               n_styles=("A", "size"))
    g["B"] = g.sumA2 / g.sumA
    s = s.join(g[["B", "n_styles"]], on="group")
    s["S"] = s.A / s.B
    s["label"] = label
    return s.sort_values("S", ascending=False)


seasons = {
    "Summer19": season("2019-06-30", "2019-09-22", "Summer19"),
    "Autumn19": season("2019-09-23", "2019-12-15", "Autumn19"),
    "Summer20": season("2020-06-30", "2020-09-22", "Summer20"),
}

show = ["style_id", "S", "F", "revenue", "units", "articles", "n_styles"]
for name, s in seasons.items():
    print(f"\n================ {name}: {len(s):,} styles in {s.group.nunique()} groups")
    t = s.head(15)[show].copy(); t["revenue"] = t.revenue.round(0)
    print("TOP 15 by success score S"); print(t.round(3).to_string(index=False))
    t = s.sort_values("revenue", ascending=False).head(10)[show].copy(); t["revenue"] = t.revenue.round(0)
    print("\nfor comparison, TOP 10 by raw revenue"); print(t.round(3).to_string(index=False))
    top50 = s.head(50)
    print(f"\ntop-50 diversity: {top50.group.nunique()} groups, "
          f"{top50.style_id.str.contains('Solid').sum()} 'Solid', "
          f"{top50.style_id.str.contains('Black').sum()} 'Black', "
          f"median articles={top50.articles.median():.0f}, median units={top50.units.median():.0f}")
    # does the score depend on number of styles in group? (should be ~0)
    gs = s.groupby("group").agg(maxS=("S", "max"), n=("n_styles", "first"))
    gs = gs[gs.n >= 5]
    print(f"Spearman(group max S, #styles in group) = {gs.maxS.corr(gs.n, method='spearman'):.2f}")
    sh = s.revenue / s.groupby("group").revenue.transform("sum")
    gsh = sh.groupby(s.group).max().loc[gs.index]
    print(f"Spearman(group max SHARE, #styles in group) = {gsh.corr(gs.n, method='spearman'):.2f}  (for contrast)")

# ---------- stability across seasons ----------
def compare(a, b):
    m = seasons[a][["style_id", "S"]].merge(seasons[b][["style_id", "S"]], on="style_id", suffixes=("_a", "_b"))
    top_a = set(seasons[a].head(50).style_id); top_b = set(seasons[b].head(50).style_id)
    print(f"{a} → {b}: Spearman(S) over {len(m):,} common styles = {m.S_a.corr(m.S_b, method='spearman'):.2f}; "
          f"top-50 overlap = {len(top_a & top_b)}/50")

print("\n================ stability")
compare("Summer19", "Autumn19")
compare("Summer19", "Summer20")
compare("Autumn19", "Summer20")
