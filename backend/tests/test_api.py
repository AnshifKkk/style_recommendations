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


def test_local_images_are_served(tmp_path):
    from backend.api import create_app
    from backend.model_service import StyleService
    service = StyleService(images_dir=tmp_path, image_url_template=None)
    style_id = service.top(1)["styles"][0]["style_id"]
    article_id = service.get(style_id)["articles"][0]["article_id"]
    (tmp_path / style_id).mkdir()
    (tmp_path / style_id / f"{article_id}.jpg").write_bytes(b"\xff\xd8\xff")   # JPEG magic bytes as a stand-in image
    client = TestClient(create_app(service))
    url = client.get("/styles/top?limit=1").json()["styles"][0]["image_url"]
    assert url == f"/images/{style_id}/{article_id}.jpg"
    assert client.get(url).status_code == 200


def test_unknown_or_unavailable_images_return_404(tmp_path):
    from backend.api import create_app
    from backend.model_service import StyleService
    service = StyleService(images_dir=tmp_path, image_url_template=None, images_on_demand=False)
    style_id = service.top(1)["styles"][0]["style_id"]
    article_id = service.get(style_id)["articles"][0]["article_id"]
    client = TestClient(create_app(service))
    assert client.get(f"/images/{style_id}/{article_id}.jpg").status_code == 404        # known, but not on disk
    assert client.get(f"/images/{style_id}/0000000000.jpg").status_code == 404          # not an image the app shows
    assert client.get(f"/images/{style_id}/../../README.md").status_code == 404


def test_image_url_template():
    from backend.model_service import StyleService
    service = StyleService(image_url_template="https://cdn.example.com/{prefix}/{article_id}.jpg")
    article = service.get(service.top(1)["styles"][0]["style_id"])["articles"][0]
    assert article["image_url"] == f"https://cdn.example.com/{article['article_id'][:3]}/{article['article_id']}.jpg"


def test_seasons_timeline(client):
    seasons = client.get("/seasons").json()
    statuses = [s["status"] for s in seasons]
    assert statuses.count("predicted") == 1
    first_predicted = statuses.index("predicted")
    assert set(statuses[:first_predicted]) == {"actual"} and set(statuses[first_predicted + 1:]) == {"coming_soon"}


def test_past_season_shows_actual_top3_with_articles(client):
    past = next(s for s in client.get("/seasons").json() if s["status"] == "actual")
    body = client.get(f"/seasons/{past['quarter_id']}").json()
    assert [s["rank"] for s in body["styles"]] == [1, 2, 3]
    scores = [s["success_score"] for s in body["styles"]]
    assert scores == sorted(scores, reverse=True)
    for style in body["styles"]:
        assert len(style["articles"]) == 5 and style["concept"] is None
        assert "as much as a typical" in style["score_explanation"]


def test_predicted_season_has_concept_slot(client):
    predicted = next(s for s in client.get("/seasons").json() if s["status"] == "predicted")
    body = client.get(f"/seasons/{predicted['quarter_id']}").json()
    assert len(body["styles"]) == 3
    assert all(s["concept"] is not None and "(predicted)" in s["score_explanation"] for s in body["styles"])


def test_coming_soon_and_unknown_seasons(client):
    coming = next(s for s in client.get("/seasons").json() if s["status"] == "coming_soon")
    assert client.get(f"/seasons/{coming['quarter_id']}").json()["styles"] == []
    assert client.get("/seasons/2035_spring").status_code == 404
    assert client.get("/seasons/not_a_season").status_code == 404
