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
- **Strong seasonality.** A style's success correlates with the same dates a year later (median +0.56 for the
  top 300 styles) and negatively with the opposite half-year (−0.41). So "same quarter last year" is useful history.
- **Short enough for the data.** The two years contain 8 quarters, 4 of which have the same quarter a year earlier.
  Half-year seasons would leave only 4 periods, with just 2 of them having a year-earlier equivalent.

The 22 days of September 2020 in the data belong to the target quarter. They are not used as model inputs.

## Limitations

- **No stock data.** Sales measure demand only where items were available. A sold-out style looks weaker
  than it was, and the success score can't tell supply apart from demand (see flag above).
- Prices are scaled by Kaggle, so only comparisons between prices are meaningful.
- **A style counts in its own group benchmark.** A strong style raises its category's benchmark and so
  lowers its own score; in the extreme, a style that dominates its category scores ≈ 1. We tested a
  leave-one-out benchmark (built only from the *other* styles in the category). It removed this effect,
  but categories with only 2–3 styles produced extreme scores (> 100), and even with size floors the
  winners became "default colour" categories (black belts, tights, leggings, ballerinas). So we kept the
  inclusive benchmark for now. Single-style categories are negligible either way (35–49 per season,
  ≈ 0.1% of revenue).

## Project structure

```
data_science/
  data.py                     # loads the raw CSVs (builds a Parquet cache of transactions in data/)
  features/
    style.py                  # style definition and style_id
    success.py                # success score
  notebooks/
    01_style_and_success_definition.ipynb
backend/  frontend/  models/  outputs/   # to be filled
```

_TBD: model training, backend, frontend, generated concepts._

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows (use .venv/bin/pip on macOS/Linux)
.venv/Scripts/jupyter notebook data_science/notebooks/
```
