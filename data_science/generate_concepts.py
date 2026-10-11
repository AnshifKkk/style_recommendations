"""Generate a next-season product concept for each predicted winning style.

Two free services, one step each:
  1. Design (Gemini text model, sees images): from the season, the style's attributes and its 5 best-selling articles
     (photos attached + their descriptions), Gemini designs one new product and returns a very detailed specification
     (silhouette, length, fit, waist/closure, fabric, texture, construction, details, hem, colour), what it keeps from the
     best sellers, what it changes, and a ready-to-render image prompt.
  2. Render (Cloudflare Workers AI, FLUX.1 schnell, text-to-image): turns that prompt into a catalogue photo.
The model's success factors are not part of the prompt; they go into the explanation that links
Predicted Style → Key Success Factors → New Product Concept.

    python -m data_science.generate_concepts           # generate concepts that don't exist yet, then assemble
    python -m data_science.generate_concepts --force   # regenerate all

Credentials (environment variables, or lines in the project's git-ignored .env file):
    GEMINI_API_KEY                                   Google AI Studio (free tier is enough for the text model)
    CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN      Cloudflare Workers AI (free plan: 10,000 neurons/day)

Writes
- outputs/concepts/<style_id>.json         the design specification from Gemini
- outputs/concepts/<style_id>.png          the rendered concept
- outputs/concepts/concepts.json           style_id → concept summary (shown in the app)
- outputs/generated_concepts.png           the three concepts in one image
- outputs/concepts_explanation.md          Predicted Style → Key Success Factors → New Product Concept
"""
import argparse
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from data_science.data import ROOT
from data_science.features.labels import FEATURE_LABELS, LOG_FEATURES
from data_science.image_source import ImageSourceUnavailable, connect, download

OUTPUTS = ROOT / "outputs"
CONCEPTS_DIR = OUTPUTS / "concepts"
IMAGES_DIR = OUTPUTS / "images"
TEXT_MODELS = os.environ.get("GEMINI_TEXT_MODELS", "gemini-3.8-flash,gemini-3.5-flash").split(",")  # fallback if busy
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
IMAGE_STEPS = 8                          # FLUX.1 schnell allows up to 8; more steps = more detail
PHOTO_STYLE = ("flat lay e-commerce catalogue photograph of the single garment, photographed from directly above, "
               "on a plain light grey background, soft even studio lighting, sharp fabric detail, no person, no model, "
               "no mannequin, no other items, no text, no logo")


# ------------------------------------------------------------------ inputs
def env(name: str) -> str | None:
    """An environment variable, or the same key in the project's .env file."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def _display(feature: str, raw: str) -> str:
    try:
        value = float(raw)
    except ValueError:
        return raw
    if feature in LOG_FEATURES:
        value = np.expm1(value)
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2f}"


def style_inputs() -> list[dict]:
    """The predicted winning styles with their attributes, best sellers and (for the explanation) success factors."""
    winners = pd.read_parquet(OUTPUTS / "season_winners.parquet")
    predicted = winners[winners.status == "predicted"].sort_values("rank")
    quarter_id = predicted.quarter_id.iloc[0]
    year, season = quarter_id.split("_")
    attributes = pd.read_csv(OUTPUTS / "predictions.csv").set_index("style_id")
    explanations = pd.read_parquet(OUTPUTS / "explanations.parquet")
    articles = pd.read_parquet(OUTPUTS / "season_articles.parquet")

    styles = []
    for w in predicted.itertuples():
        a = attributes.loc[w.style_id]
        drivers = explanations[(explanations.style_id == w.style_id) & (explanations.contribution > 0)]
        best = articles[(articles.style_id == w.style_id) & (articles.quarter_id == quarter_id)].sort_values("units", ascending=False)
        styles.append({
            "rank": int(w.rank), "style_id": w.style_id, "style_name": w.style_name, "season": f"{season.title()} {year}",
            "predicted_score": round(float(w.success_score), 2),
            "attributes": {"segment": a.index_group_name, "product type": a.product_type_name,
                           "pattern": a.graphical_appearance_name, "colour": a.colour_group_name,
                           "tone": a.perceived_colour_value_name},
            "success_factors": [f"{FEATURE_LABELS.get(d.feature, d.feature)}: {_display(d.feature, d.feature_value)}"
                                for d in drivers.head(4).itertuples()],
            "best_sellers": [{"article_id": f"{int(r.article_id):010d}", "name": r.prod_name,
                              "description": None if pd.isna(r.detail_desc) else r.detail_desc, "units": int(r.units)}
                             for r in best.itertuples()],
        })
    return styles


def reference_images(style: dict) -> list[Image.Image]:
    """All best sellers' photos, fetched from Kaggle first if they aren't on disk yet."""
    images, api = [], None
    for b in style["best_sellers"]:
        path = IMAGES_DIR / style["style_id"] / f"{b['article_id']}.jpg"
        if not path.exists():
            try:
                api = api or connect()
                download(api, b["article_id"], path)
            except (ImageSourceUnavailable, Exception) as error:
                print(f"   reference image {b['article_id']} unavailable ({type(error).__name__}), continuing without it")
                continue
        images.append(Image.open(path).convert("RGB"))
    return images


# ------------------------------------------------------------------ step 1: design (Gemini, sees the photos)
def design_prompt(style: dict) -> str:
    a = style["attributes"]
    sellers = "\n".join(f"{i}. {b['name']}: {b['description']}" for i, b in enumerate(style["best_sellers"], 1))
    product = a["product type"].lower()
    return f"""You are a senior fashion designer at a high-street retailer, designing a new product for the {style['season']} collection.

STYLE TO DESIGN FOR: {a['segment']} · {a['product type']} · {a['pattern']} · {a['colour']} · {a['tone']} tone

The attached photos are this style's best-selling articles, in this order:
{sellers}

Study every photo and description closely: silhouette, length, fit, waistband or neckline, closures, seams, fabric,
texture, weight and drape, hems and finishing details. Identify the visual ideas they share.

Then design ONE new {product} for {style['season']}:
- Stay within the style: {a['segment']}, {product}, {a['pattern'].lower()}, {a['colour'].lower()}, {a['tone'].lower()} tone.
- KEEP the strongest visual ideas the best sellers share.
- CHANGE something meaningful for {style['season']} (silhouette, length, construction details, fabric or texture), so it
  is a genuinely new product and not a copy of any best seller.
- Keep it realistic to produce at high-street prices.

Be very specific and concrete: a pattern maker must be able to draw it, and an image generator must be able to render it
exactly. Use measurable or visual terms (e.g. "hem 8 cm below the knee", "5 cm wide rib-knit waistband"), not vague ones.

Answer as JSON with exactly these keys:
"name": 2-4 word product name,
"keeps": list of 2-4 visual ideas taken from the best sellers, each saying which best seller(s) it comes from,
"changes": list of 2-4 design changes for {style['season']}, each saying why it suits the season,
"description": one sentence for a merchandiser,
"design_details": object with these keys, each 1-3 detailed sentences:
    "silhouette_and_fit", "length_and_proportions", "waist_neckline_and_closures", "fabric_and_weight",
    "texture_and_surface", "construction_and_seams", "details_and_trims", "hem_and_finishing", "colour_and_finish",
"visual_description": a 150-250 word paragraph describing exactly what the garment looks like, from top to bottom,
"image_prompt": a 60-110 word prompt for a text-to-image model that describes the garment's look precisely (start with
    the colour and garment type, name every visible detail) and ends with: "{PHOTO_STYLE}"."""


def design(client, style: dict, refs: list[Image.Image]) -> dict:
    """Gemini's design specification. Tries each text model in turn; retries when Google is busy."""
    from google.genai import types
    contents = [design_prompt(style), *refs]
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.8)
    last_error = None
    for model in TEXT_MODELS:
        for attempt in range(4):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                return {**json.loads(response.text), "text_model": model}
            except Exception as error:
                last_error, code = error, getattr(error, "code", None)
                if not (code == 429 or (isinstance(code, int) and code >= 500)):
                    raise
                print(f"   {model} busy ({code}), retrying in {15 * (attempt + 1)}s")
                time.sleep(15 * (attempt + 1))
    raise last_error


# ------------------------------------------------------------------ step 2: render (Cloudflare FLUX.1 schnell)
def render(prompt: str, account_id: str, token: str) -> Image.Image:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{IMAGE_MODEL}"
    for attempt in range(3):
        response = httpx.post(url, headers={"Authorization": f"Bearer {token}"}, timeout=120,
                              json={"prompt": prompt[:2048], "steps": IMAGE_STEPS})
        if response.status_code == 200:
            return Image.open(BytesIO(base64.b64decode(response.json()["result"]["image"]))).convert("RGB")
        if response.status_code not in (429, 500, 502, 503) or attempt == 2:
            raise RuntimeError(f"Cloudflare {response.status_code}: {response.text[:300]}")
        time.sleep(10 * (attempt + 1))


# ------------------------------------------------------------------ outputs
def _font(size: int, bold: bool = False):
    for name in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(draw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) <= width:
            line = candidate
        else:
            lines.append(line)
            line = word
    return lines + [line]


def compose(styles: list[dict], designs: dict, path: Path) -> None:
    """The three concepts side by side, each with its concept name, predicted style and score."""
    panel_w, img_h, pad, caption_h, title_h = 600, 600, 40, 230, 110
    canvas = Image.new("RGB", (pad + len(styles) * (panel_w + pad), title_h + img_h + caption_h + pad), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 30), f"Next-season concepts · {styles[0]['season']}", fill="#171717", font=_font(44, bold=True))
    draw.text((pad, 82), "One AI-generated concept for each predicted winning style", fill="#737373", font=_font(22))
    for i, s in enumerate(styles):
        x = pad + i * (panel_w + pad)
        concept = Image.open(CONCEPTS_DIR / f"{s['style_id']}.png").convert("RGB")
        concept.thumbnail((panel_w, img_h))
        canvas.paste(concept, (x + (panel_w - concept.width) // 2, title_h + (img_h - concept.height) // 2))
        y = title_h + img_h + 20
        draw.text((x, y), f"#{s['rank']}  {designs[s['style_id']]['name']}", fill="#e60023", font=_font(30, bold=True))
        y += 44
        for line in _wrap(draw, s["style_name"], _font(22), panel_w):
            draw.text((x, y), line, fill="#171717", font=_font(22))
            y += 30
        draw.text((x, y + 4), f"Predicted success score {s['predicted_score']:.2f}", fill="#737373", font=_font(20))
    canvas.save(path)


def write_explanation(styles: list[dict], designs: dict, path: Path) -> None:
    out = [f"# Next-season concepts · {styles[0]['season']}", "",
           "![Generated concepts](generated_concepts.png)", "",
           "Each concept follows **Predicted Style → Key Success Factors → New Product Concept**. Gemini designed each "
           "concept from the style's attributes and its 5 best-selling articles (photos and descriptions); FLUX.1 schnell "
           "(Cloudflare Workers AI) rendered it. The success factors come from the prediction model.", ""]
    for s in styles:
        d = designs[s["style_id"]]
        out += [f"## #{s['rank']} {s['style_name']}", "",
                f"**Predicted style:** {s['style_name']}, predicted success score {s['predicted_score']:.2f} "
                f"(≈ {s['predicted_score']:.1f}× a typical {s['attributes']['segment']} {s['attributes']['product type'].lower()} style).", "",
                "**Key success factors** (largest contributions to the prediction):", *[f"- {f}" for f in s["success_factors"]],
                f"- Best sellers last quarter: {', '.join(b['name'] for b in s['best_sellers'])}", "",
                f"**New product concept: {d['name']}**", "", d["description"], "",
                f"![{d['name']}](concepts/{s['style_id']}.png)", "",
                "Keeps:", *[f"- {k}" for k in d["keeps"]], "", "Changes:", *[f"- {c}" for c in d["changes"]], "",
                "<details><summary>Full design specification</summary>", "",
                *[f"- **{k.replace('_', ' ').capitalize()}:** {v}" for k, v in d["design_details"].items()], "",
                d["visual_description"], "", "</details>", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate designs and images that already exist")
    args = parser.parse_args()

    gemini_key = env("GEMINI_API_KEY")
    account_id, token = env("CLOUDFLARE_ACCOUNT_ID"), env("CLOUDFLARE_API_TOKEN")
    if not gemini_key:
        raise SystemExit("GEMINI_API_KEY missing (environment or .env)")

    from google import genai
    client = genai.Client(api_key=gemini_key)
    CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)
    styles = style_inputs()
    designs = {}

    for s in styles:
        spec_path, image_path = CONCEPTS_DIR / f"{s['style_id']}.json", CONCEPTS_DIR / f"{s['style_id']}.png"
        print(f"#{s['rank']} {s['style_name']}")
        if spec_path.exists() and not args.force:
            designs[s["style_id"]] = json.loads(spec_path.read_text(encoding="utf-8"))
        else:
            refs = reference_images(s)
            designs[s["style_id"]] = {**design(client, s, refs), "reference_articles": [b["article_id"] for b in s["best_sellers"]],
                                      "prompt": design_prompt(s)}
            spec_path.write_text(json.dumps(designs[s["style_id"]], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   design: {designs[s['style_id']]['name']}")

        if image_path.exists() and not args.force:
            print("   image: exists")
        elif account_id and token:
            render(designs[s["style_id"]]["image_prompt"], account_id, token).save(image_path)
            print(f"   image: {image_path.relative_to(ROOT)}")

    missing = [s for s in styles if not (CONCEPTS_DIR / f"{s['style_id']}.png").exists()]
    if missing:
        raise SystemExit(f"designs saved; {len(missing)} image(s) not rendered: set CLOUDFLARE_ACCOUNT_ID and "
                         "CLOUDFLARE_API_TOKEN (environment or .env) and run again")

    (CONCEPTS_DIR / "concepts.json").write_text(json.dumps(
        {sid: f"{d['name']}: {d['description']}" for sid, d in designs.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    compose(styles, designs, OUTPUTS / "generated_concepts.png")
    write_explanation(styles, designs, OUTPUTS / "concepts_explanation.md")
    print(f"wrote {(OUTPUTS / 'generated_concepts.png').relative_to(ROOT)} and {(OUTPUTS / 'concepts_explanation.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
