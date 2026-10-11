# Next-Season Style Intelligence Platform

Predict the fashion styles most likely to perform strongly next season, turn them into new product
concepts with generative AI, and let business users explore the results through an API and a web app.

```
Retail Data → Data Processing → ML Model → Prediction Output → Backend API → Frontend → Business User
```

> 🚧 Work in progress. Sections marked _TBD_ will be filled in as the project develops.

## Data

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (Kaggle).

| File | Used | Purpose |
|---|---|---|
| `transactions_train.csv` | ✅ | 31.8M rows, one row per unit sold (2018-09-20 → 2020-09-22): date, customer, article, scaled price, channel |
| `articles.csv` | ✅ | 105,542 articles with product attributes; used to define styles |
| images | ✅ (subset) | Visual reference for the top styles and concept generation |
| `customers.csv` | ❌ | Not required by the brief; customer behaviour is derived from `customer_id` in transactions |
| `sample_submission.csv` | ❌ | Belongs to the original recommendation competition |

Place the raw CSVs in `kaggle_downlods/` (git-ignored).

### Data subset

- **Transactions: all of them, no sampling.** The data isn't too big for a laptop (16 GB RAM): as a Parquet
  cache the transactions take 0.77 GB on disk and ~0.7 GB in memory, and load in seconds. Using everything avoids
  sampling bias. The days 1–22 Sep 2020 belong to the prediction quarter and are not used as model input.
- **Cleaning: 376 articles removed, together with their transactions (99.6% of revenue kept).**
  - Non-fashion product groups: Items, Cosmetic, Furniture, Garment and Shoe care, Interior textile, Stationery,
    Fun (98 articles, ~0.01% of revenue). They have no style to predict.
  - Articles with an unknown/undefined value in a style attribute (278 articles, ~0.4% of revenue). Their style isn't defined.
  - This leaves 105,166 articles and 31,653,188 transactions. Implemented in `data_science/data.py` (`load_articles`).
  - Kept on purpose: identical transaction rows (~9%, multiple units, since there is no quantity column), very low prices
    (~0.01% of revenue) and bulk purchases (~0.2% of revenue). Details in
    [`03_data_quality_and_eda.ipynb`](data_science/notebooks/03_data_quality_and_eda.ipynb).
- **Images: only what's needed.** The full set is ~29 GB; we only use images of the top styles and of the
  articles shown in the app.
- **COVID is kept and flagged.** Store sales are missing from 2020-03-20 to 2020-05-05 (all sales are online).
  Style rankings in Spring 2020 stay about as stable as in normal quarters, because the success score compares
  styles within their category, so the quarter is kept.

## Key definitions

### What is a "style"?

A style is the set of attributes that together define a customer's style choice. It has to be:

- **Exhaustive**: changing any attribute *outside* the set doesn't change the style.
- **Non-redundant**: changing the value of any attribute *inside* the set gives a different style.

```
style = index_group + product_type + graphical_appearance + colour_group + perceived_colour_value
e.g.    Ladieswear  · Dress        · All over pattern     · Light Pink   · Dusty Light
```

| Attribute | Why it is included |
|---|---|
| `index_group_name` | Who it is for (Ladieswear / Menswear / Divided / Baby-Children / Sport). 80% of articles would otherwise be in styles that mix different customer segments |
| `product_type_name` | What it is (dress, trousers, hoodie, ...) |
| `graphical_appearance_name` | Surface design: pattern and texture (solid, stripe, denim, lace, ...) |
| `colour_group_name` | The specific colour (50 values) |
| `perceived_colour_value_name` | Tone (dark / light / dusty light / bright). Splits the same colour into meaningfully different looks, e.g. Red is 49% Medium / 47% Bright |

Excluded: identifiers (`article_id`, `product_code`, `prod_name`); columns that duplicate an included one
(`product_group_name`, `index_name`, `perceived_colour_master_name`); and internal merchandising
structure (`department_name`, `section_name`, `garment_group_name`).

**Why not articles or product codes?** Most articles are replaced every season, so next season's
articles mostly don't exist yet. Attribute-based styles recur across seasons and have history to learn from.

**Known gap:** fit, length, neckline and fabric only exist in free text (`detail_desc`) and images,
so they are not part of the style.

### What is a "successful" style?

A style is successful if it earns strong full-price revenue **relative to the demand of its category**.
Category = `index_group + product_type`.

```
Adjusted revenue = Revenue × (Revenue ÷ Full-price revenue)
Group benchmark  = Σ Adjusted revenue² ÷ Σ Adjusted revenue      (over styles in the same category)
Success score    = Adjusted revenue ÷ Group benchmark
```

- **Full-price revenue** is what the units would have earned at each article's highest observed price,
  so discounting lowers the score.
- **Group benchmark** is the revenue-weighted typical style size of the category. Styles are weighted by
  revenue, so a long tail of tiny styles barely affects it. A score of 1 means "as good as the typical
  style a sale in this category comes from".
- Styles are ranked by success score. There is no minimum-size floor: a small style that beats its category
  performed well. Whether that success repeats is left to the prediction model (`articles` will be a feature).

Checked on Summer 2019, Autumn 2019 and Summer 2020
([`01_style_and_success_definition.ipynb`](data_science/notebooks/01_style_and_success_definition.ipynb)):

- Raw revenue puts only Ladieswear basics in the top 10. The success score's top 50 comes from 45–48 different categories.
- Scores are fairly stable from season to season (rank correlation 0.51–0.75; about half of the top 50 carries over), so the problem is predictable but not trivial.
- Very small styles are rare at the top: 1–3 of the top 50 per season have fewer than 5 articles.

### ⚠️ Flag: supply vs. demand

The top styles in each category are mostly solid black/white basics. Part of that comes from H&M's own
supply decisions: for example, H&M offered 104 black solid skirts versus a handful in most other colours,
so they naturally earned more in total. **The success score therefore partly reflects H&M's buying plan,
not only what customers prefer.** We accept this deliberately: these styles really did perform best,
and predicting what performs is the goal. Without stock/assortment data we can't fully separate the two effects.

### Prediction period

**Autumn 2020: 1 Sep – 30 Nov 2020**, predicted from data up to 31 Aug 2020.

We predict the best three styles for **one quarter**, assuming production and restocking decisions are made
quarterly. The success score is aggregated over the whole quarter, because that is the level at which
"what should we produce" is decided.

Quarters follow the fashion seasons: Spring (Mar–May), Summer (Jun–Aug), Autumn (Sep–Nov), Winter (Dec–Feb).
They line up with the half-year switch points seen in the data (styles' success changes around 1 March and
1 September), while calendar quarters would mix summer and autumn styles.

Evidence ([`02_eda_trends.ipynb`](data_science/notebooks/02_eda_trends.ipynb)):

- **Long enough to be reliable.** Scoring each style separately from the odd and even days of the same period,
  the two rankings agree at 0.97 for ~91-day periods (top-50 overlap 0.90). Noise only matters below about a month.
- **Strong seasonality.** A style's success correlates with the same dates a year later (median +0.55 for the
  top 300 styles) and negatively with the opposite half-year (−0.43). So "same quarter last year" is useful history.
- **Short enough for the data.** The two years contain 8 quarters, 4 of which have the same quarter a year earlier.
  Half-year seasons would leave only 4 periods, with just 2 of them having a year-earlier equivalent.

The 22 days of September 2020 in the data belong to the target quarter. They are not used as model inputs.

## Modelling approach

### The problem

- **Unit:** one row per style per quarter
  ([`04_style_quarter_table.ipynb`](data_science/notebooks/04_style_quarter_table.ipynb)).
- **Target:** that quarter's success score.
- **Output:** a ranking of styles for the next quarter; the top 3 are the winners.
- **Shape:** three properties decide the model:
  - **Very short history per style:** at most 8 quarters, i.e. two seasonal cycles.
  - **Many related styles:** ~23k styles share categories, colours, patterns and seasons, so a model can learn across styles.
  - **Only the top of the ranking matters:** the target is heavily skewed (90% of style-quarters score below 0.25, the best ~4–5)
    and about half of the rows are zero. Business users also need an explanation for every prediction.

### Models considered

| Family | Verdict | Reason |
|---|---|---|
| Baseline: last quarter's success score | **Baseline** | The simplest forecast: "next quarter looks like last quarter". Success persists between quarters, so it is a real bar to beat. |
| Per-style time series (ARIMA, ETS, Prophet, Croston) | Rejected | See below. |
| Deep learning (DeepAR, N-BEATS, Temporal Fusion Transformer, tabular MLP/TabNet) | Rejected | See below. |
| Pretrained time-series models (Chronos, TimesFM) | Rejected | Forecast a single numeric series; our target is a relative score that depends on the other styles, and each series has ≤ 8 points. |
| **Ridge regression** on engineered features, one model across all styles | **Final model** | Linear, regularised (stable with correlated features such as the different lags), fast, and every prediction decomposes exactly into feature contributions. |
| **LightGBM** (gradient-boosted trees) on the same features | Compared | The standard non-linear model for tabular retail forecasting. Tests whether interactions and non-linear effects add anything. A learning-to-rank variant was also tried and ranked worse at the top. |
| Hierarchical / Bayesian panel model | Future improvement | Shrinks small styles towards their category; interpretable, but slower to build. |
| Image embeddings (CLIP/CNN) as features | Future improvement | Could capture fit and silhouette, which the attributes miss; needs the ~29 GB image set. |
| Customer-level signals (early adopters, purchase embeddings) | Future improvement | Styles bought early by "trend-leading" customers may rise next; needs customer-level modelling. |

**Why not per-style time series.** Classical models (ARIMA, ETS, Prophet) fit one series at a time. Each style has
at most 8 quarterly points, and with a seasonal period of 4 that is only two cycles: too few to estimate trend
*and* seasonality without fitting noise. They also learn nothing from the other 23k styles, while the strongest
signals here are shared (e.g. "black solid basics persist", "sweaters rise in autumn"). New and young styles,
with 1–3 quarters of history, can't be modelled at all.

**Why not deep learning.** Sequence models (DeepAR, N-BEATS, TFT) learn patterns along time and need long series,
typically dozens to hundreds of steps per series; we have 8. Moving to weekly data would give ~100 steps, but the
decision and the target are quarterly, the models are data-hungry and slow to tune on a laptop, and their
predictions are hard to explain to business users, which the brief requires. Tabular deep learning
(MLP, TabNet) rarely beats simpler models on ~100k rows. The limiting factor is the length of history, which
no architecture can add.

**What we do instead.** Turn each style's history into features (lags, same quarter last year, recent weeks,
seasonal share, category trends, attributes) and train **one model across all styles**. A short history becomes
usable because the model learns from 145k style-quarters at once.

### Features

All features for a quarter use only data from before the quarter starts (`data_science/features/build.py`):

| Group | Features |
|---|---|
| Style history | success score 1–4 quarters ago (4 = same quarter last year), 4-quarter mean / max / trend, quarters sold, age, time since last sale, last quarter's revenue, units, articles and full-price factor |
| Recent momentum | success score in the last 4 weeks, revenue momentum, new articles launched in the last 13 weeks |
| Seasonality | target season; share of the style's and its category's past revenue earned in that season |
| Attributes | segment, product type, pattern, colour, tone (categorical) |
| Attribute momentum | is the style's colour / pattern / tone gaining share of all revenue, year on year? |
| Category context | category growth, number of competing styles, category benchmark |
| Customers and price | buyers last quarter, repeat-purchase rate, online share, price relative to the category |

### Evaluation

- **Walk-forward:** for each evaluation quarter, models are trained only on earlier quarters and predict every style
  with sales history. Winter 2019 and Spring 2020 are used to choose the model; **Summer 2020 is the test quarter**.
- **Metric, top-3 quality:** the average actual success score of the three styles we pick ÷ the average actual score of
  the true top three. 1.0 means our picks did as well as the best possible picks. It measures exactly the business
  decision (which three styles to produce), and still gives credit when a pick is a near-miss.

### Results

| Model | Winter 2019 | Spring 2020 (COVID) | **Validation mean** | **Test: Summer 2020** |
|---|---|---|---|---|
| Baseline: last quarter | 0.866 | 0.717 | 0.791 | 0.839 |
| **Ridge** | **0.921** | 0.879 | **0.900** | **0.894** |
| LightGBM | 0.872 | **0.916** | 0.894 | 0.825 |

- Both models beat the baseline, most clearly in Spring 2020 when COVID disrupted last quarter's pattern.
- **Ridge is chosen**: best on validation and on the test quarter, simpler, and its explanations are exact.

**Why a linear model is enough.** The relationship between a style's recent performance and its next-quarter
performance is close to a straight line (on the log scale):

- One straight line on a single feature, the success score in the last 4 weeks, explains **65%** of the variance
  of next quarter's (log) score on the test quarter. Ridge with all features reaches **69%**, LightGBM **71%**.
  The non-linear model adds only ~2 points.
- LightGBM's learned effect of the last-4-weeks score rises steadily and then **flattens above ~1.5**: trees cannot
  extrapolate beyond the values in a leaf, so the strongest styles all get similar predictions. That is exactly where
  the top three are decided, and where a linear model, which keeps increasing, separates them better.
- Most of the signal is persistence (recent and average past success) plus seasonality, both of which are additive.

### Final model and prediction

`python -m data_science.train` evaluates and then trains ridge on all labelled quarters (`models/ridge.joblib`).
`python -m data_science.predict` scores every style for Autumn 2020 (`outputs/predictions.csv`) and writes each
prediction's largest feature contributions (`outputs/explanations.csv`). Because the model is linear, a prediction is
exactly the average prediction plus the sum of contributions, coefficient × (feature value − training average),
which equals its SHAP values. The predicted score is on the success-score scale; values above the historical maximum
reflect linear extrapolation and only the ranking is used.

## Limitations

- **No stock data.** Sales measure demand only where items were available. A sold-out style looks weaker
  than it was, and the success score can't tell supply apart from demand (see flag above).
- Prices are scaled by Kaggle, so only comparisons between prices are meaningful.
- **A style counts in its own group benchmark.** A strong style raises its category's benchmark and so
  lowers its own score; in the extreme, a style that dominates its category scores ≈ 1. We tested a
  leave-one-out benchmark (built only from the *other* styles in the category). It removed this effect,
  but categories with only 2–3 styles produced extreme scores (> 100), and even with size floors the
  winners became "default colour" categories (black belts, tights, leggings, ballerinas). So we kept the
  inclusive benchmark for now. Single-style categories are negligible either way (≈ 0.1% of revenue).

## Project structure

```
data_science/
  data.py                     # loads and cleans the raw CSVs (Parquet cache of transactions in data/)
  features/
    style.py                  # style definition and style_id
    success.py                # success score
    quarters.py               # fashion quarters and the style × quarter table
    build.py                  # feature engineering
  train.py                    # walk-forward evaluation + final model training
  predict.py                  # scores the prediction quarter, with explanations
  notebooks/
    01_style_and_success_definition.ipynb   # style and success metric decisions
    02_eda_trends.ipynb                     # how style success changes over time → prediction period
    03_data_quality_and_eda.ipynb           # data quality, cleaning, sales/product/category/customer EDA
    04_style_quarter_table.ipynb            # the style × quarter table and its target
models/ridge.joblib           # final model
outputs/                      # evaluation.csv, predictions.csv, explanations.csv
backend/  frontend/           # to be filled
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows (use .venv/bin/pip on macOS/Linux)
.venv/Scripts/python -m data_science.train      # evaluate + train the final model
.venv/Scripts/python -m data_science.predict    # predict Autumn 2020
```
