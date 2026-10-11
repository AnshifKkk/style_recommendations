"""Response models of the API (they also generate the OpenAPI documentation at /docs)."""
from typing import Literal

from pydantic import BaseModel, Field


class QuarterSales(BaseModel):
    quarter_id: str = Field(examples=["2020_summer"])
    season: str
    success_score: float = Field(description="Success score of the style in that quarter (0 = no sales)")
    revenue: float = Field(description="Revenue in the quarter (scaled prices, comparable between styles only)")
    units: int
    articles: int = Field(description="Number of different articles of the style that sold")


class StyleSummary(BaseModel):
    rank: int | None = Field(description="Rank among commercially relevant styles; null if the style is not eligible")
    style_id: str = Field(examples=["ladieswear__skirt__solid__black__dark"])
    name: str = Field(examples=["Ladieswear · Skirt · Solid · Black · Dark"])
    category: str = Field(description="Customer segment and product type", examples=["Ladieswear · Skirt"])
    prediction_score: float = Field(description="Predicted success score for the prediction quarter (1 = as good as a "
                                                "typical style of its category). Values above the historical maximum "
                                                "(~4.75) come from the model extrapolating; only the ranking is validated.")
    relative_score: float = Field(description="prediction_score ÷ that of the #1 style (1.0 = the top style)")
    percentile: float | None = Field(description="Share of the other ranked styles this style is ahead of (1.0 = #1, 0.998 = top 0.2%); "
                                                  "null if not eligible for the ranking")
    units_last_quarter: int
    image_url: str | None = Field(description="Image of the style's best-selling article, if available locally")
    sales_history: list[QuarterSales]


class TopStylesResponse(BaseModel):
    quarter: str = Field(examples=["2020_autumn"])
    candidates: int = Field(description="Number of commercially relevant styles that were ranked")
    styles: list[StyleSummary]


class StyleList(BaseModel):
    quarter: str
    total: int
    limit: int
    offset: int
    styles: list[StyleSummary]


class Attributes(BaseModel):
    segment: str
    product_type: str
    pattern: str
    colour: str
    tone: str


class Article(BaseModel):
    article_id: str = Field(examples=["0768921001"])
    name: str
    description: str | None
    units_last_quarter: int
    image_url: str | None
    kaggle_image_path: str = Field(examples=["images/076/0768921001.jpg"])


class Contribution(BaseModel):
    feature: str
    label: str
    value: str
    contribution: float = Field(description="Effect on the prediction, in log(1 + success score)")
    effect: Literal["raises", "lowers"]


class Explanation(BaseModel):
    summary: str
    contributions: list[Contribution]


class StyleDetail(BaseModel):
    style_id: str
    name: str
    category: str
    quarter: str
    rank: int | None
    commercially_relevant: bool
    prediction_score: float = Field(description="Predicted success score for the prediction quarter (1 = as good as a "
                                                "typical style of its category). Values above the historical maximum "
                                                "(~4.75) come from the model extrapolating; only the ranking is validated.")
    relative_score: float = Field(description="prediction_score ÷ that of the #1 style (1.0 = the top style)")
    percentile: float | None = Field(description="Share of the other ranked styles this style is ahead of (1.0 = #1, 0.998 = top 0.2%); "
                                                  "null if not eligible for the ranking")
    attributes: Attributes
    articles: list[Article]
    explanation: Explanation
    sales_history: list[QuarterSales]


class Health(BaseModel):
    status: Literal["ok"]
    quarter: str
    styles: int
