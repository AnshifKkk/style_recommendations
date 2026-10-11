"""API tests against the real artifacts in outputs/ (run `python -m data_science.predict` first).

    pytest backend/tests
"""
import pytest
from fastapi.testclient import TestClient

from backend.model_service import OUTPUTS_DIR

ARTIFACTS = ["predictions.csv", "explanations.parquet", "style_history.parquet", "style_articles.parquet"]
pytestmark = pytest.mark.skipif(not all((OUTPUTS_DIR / f).exists() for f in ARTIFACTS),
                                reason="model artifacts missing: run python -m data_science.predict")


@pytest.fixture(scope="module")
def client():
    from backend.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def top(client):
    return client.get("/styles/top").json()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["styles"] > 0


def test_top_returns_three_ranked_styles(top):
    styles = top["styles"]
    assert len(styles) == 3
    assert [s["rank"] for s in styles] == [1, 2, 3]
    scores = [s["prediction_score"] for s in styles]
    assert scores == sorted(scores, reverse=True)
    assert styles[0]["relative_score"] == 1.0 and styles[0]["percentile"] == 1.0
    for s in styles:
        assert {"style_id", "prediction_score", "category", "sales_history"} <= s.keys()
        assert len(s["sales_history"]) > 0


def test_top_limit(client):
    assert len(client.get("/styles/top?limit=10").json()["styles"]) == 10


@pytest.mark.parametrize("limit", [0, 51, "abc"])
def test_top_rejects_invalid_limit(client, limit):
    assert client.get(f"/styles/top?limit={limit}").status_code == 422


def test_style_detail(client, top):
    style_id = top["styles"][0]["style_id"]
    body = client.get(f"/styles/{style_id}").json()
    assert body["style_id"] == style_id and body["rank"] == 1 and body["commercially_relevant"]
    assert set(body["attributes"]) == {"segment", "product_type", "pattern", "colour", "tone"}
    assert body["articles"] and body["articles"][0]["article_id"].isdigit() and len(body["articles"][0]["article_id"]) == 10
    assert body["explanation"]["summary"].startswith("Ranked #1")
    assert body["explanation"]["contributions"]
    assert all(c["effect"] in ("raises", "lowers") for c in body["explanation"]["contributions"])


def test_unknown_style_returns_404(client):
    response = client.get("/styles/not_a_style")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_list_filters_and_pagination(client):
    page = client.get("/styles?segment=Menswear&limit=5&offset=5").json()
    assert page["limit"] == 5 and page["offset"] == 5 and len(page["styles"]) <= 5
    assert all(s["category"].startswith("Menswear") for s in page["styles"])
    assert [s["rank"] for s in page["styles"]] == sorted(s["rank"] for s in page["styles"])


def test_styles_outside_business_rule_have_no_rank(client):
    body = client.get("/styles?include_all=true&limit=200").json()
    unranked = [s for s in body["styles"] if s["rank"] is None]
    assert unranked, "expected some styles outside the business rule among the top 200"
    assert all(s["percentile"] is None for s in unranked)
