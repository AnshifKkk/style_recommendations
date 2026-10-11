"""AI assistant: answers business questions about the predictions with Gemini and function calling.

Gemini decides which tools to call; every tool reads the same StyleService the API uses, so answers are grounded in the
model's outputs instead of the language model's memory. Needs GEMINI_API_KEY (environment or the git-ignored .env file);
the free tier of the text model is enough.
"""

import os
import time

from backend.model_service import ROOT, StyleNotFound, SeasonNotFound, StyleService

MODELS = os.environ.get("ASSISTANT_MODELS", "gemini-3.8-flash,gemini-3.5-flash,gemini-3.1-flash-lite").split(",")
MAX_TOOL_CALLS = 8

SYSTEM_PROMPT = """You are the assistant of "Style Intelligence", a tool that helps a fashion retailer decide which styles to
produce next season. You answer questions from merchandisers and buyers.

What the data is:
- H&M sales from Sep 2018 to Aug 2020. A style = customer segment + product type + pattern + colour + colour tone,
  e.g. "Ladieswear · Skirt · Solid · Black · Dark".
- Seasons are quarters: Spring (Mar–May), Summer (Jun–Aug), Autumn (Sep–Nov), Winter (Dec–Feb).
- Success score: a style's revenue (adjusted for discounts) compared with a typical style of its category; 1.0 = typical,
  4 = about 4 times as successful. Past seasons show actual scores, the next quarter (Autumn 2020) predicted ones.
- Only commercially relevant styles are ranked: revenue last quarter in the top 10% of styles that sold.
- Predicted scores above ~4.8 are the model extrapolating; the ranking is what is reliable.
- Each predicted winner has an AI-generated product concept.

How to answer:
- Always use the tools to get facts. Never invent styles, numbers or reasons. If the tools don't have it, say so.
- To find a style's id, use search_styles first.
- Explain predictions with the drivers returned by get_style_details, in plain business language. Only give a reason for
  a style if you fetched its details; otherwise list it without a reason.
- Be concise: a few sentences, or a short list with "-" bullets. Plain text only: no headings, no bold, no tables.
- Stay on topic (styles, seasons, predictions, concepts). Politely decline anything else."""


def load_api_key() -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "GEMINI_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return None


class AssistantUnavailable(RuntimeError):
    pass


# ------------------------------------------------------------------ tools (compact, JSON-friendly results)
def make_tools(service: StyleService) -> list:
    def list_seasons() -> list[dict]:
        """List all seasons: past ones (status 'actual'), the predicted next quarter, and later ones ('coming_soon')."""
        return [{"quarter_id": s["quarter_id"], "label": s["label"], "status": s["status"]} for s in service.seasons()]

    def get_season_winners(quarter_id: str) -> dict:
        """The top 3 styles of a season (e.g. quarter_id '2019_summer' or '2020_autumn'): actual winners for past
        seasons, predicted winners for the next quarter, with their success score, best-selling articles and, for the
        predicted season, the AI-generated concept."""
        try:
            season = service.season(quarter_id)
        except SeasonNotFound:
            return {"error": f"unknown season '{quarter_id}'; use list_seasons"}
        return {"season": season["label"], "status": season["status"], "styles": [
            {"rank": s["rank"], "style_id": s["style_id"], "name": s["name"], "success_score": s["success_score"],
             "best_sellers": [f"{a['name']} ({a['units']:,} units)" for a in s["articles"]],
             "concept": (s["concept"] or {}).get("description")}
            for s in season["styles"]]}

    def get_top_predicted_styles(limit: int = 10) -> dict:
        """The highest-ranked styles predicted for the next quarter (limit 1-50)."""
        top = service.top(max(1, min(int(limit), 50)))
        return {"quarter": top["quarter"], "ranked_styles": top["candidates"], "styles": [
            {"rank": s["rank"], "style_id": s["style_id"], "name": s["name"], "predicted_score": s["prediction_score"]}
            for s in top["styles"]]}

    def get_style_details(style_id: str) -> dict:
        """Everything about one style: attributes, its prediction for the next quarter (score, rank, the main drivers
        of the prediction) and its success score and units sold in every past quarter."""
        try:
            d = service.get(style_id)
        except StyleNotFound:
            return {"error": f"unknown style_id '{style_id}'; use search_styles to find it"}
        return {"name": d["name"], "attributes": d["attributes"], "prediction_quarter": d["quarter"],
                "predicted_score": d["prediction_score"], "rank": d["rank"],
                "commercially_relevant": d["commercially_relevant"],
                "why": d["explanation"]["summary"],
                "drivers": [f"{c['label']} = {c['value']} ({c['effect']} the prediction)"
                            for c in d["explanation"]["contributions"][:6]],
                "history": [f"{h['quarter_id']}: score {h['success_score']}, {h['units']:,} units" for h in d["sales_history"]],
                "best_sellers_last_quarter": [a["name"] for a in d["articles"]]}

    def search_styles(text: str = "", segment: str = "", product_type: str = "", colour: str = "", limit: int = 10) -> dict:
        """Find styles in the next-quarter ranking. All filters are optional and case-insensitive: `text` matches any
        part of the style name (e.g. 'denim', 'dress pink'), `segment` is Ladieswear / Menswear / Divided / Sport /
        Baby/Children, `product_type` e.g. Dress or Trousers, `colour` e.g. Black or Light Pink. Returns ranked styles
        with their style_id and predicted score."""
        rows = service.ranked
        for column, value in (("index_group_name", segment), ("product_type_name", product_type), ("colour_group_name", colour)):
            if value:
                rows = rows[rows[column].str.lower() == value.lower()]
        for word in text.lower().split():
            rows = rows[rows.style_name.str.lower().str.contains(word, regex=False)]
        rows = rows.head(max(1, min(int(limit), 50)))
        return {"matches": len(rows), "styles": [
            {"rank": int(r["rank"]), "style_id": r.style_id, "name": r.style_name, "predicted_score": round(float(r.prediction_score), 2)}
            for _, r in rows.iterrows()]}

    return [list_seasons, get_season_winners, get_top_predicted_styles, get_style_details, search_styles]


# ------------------------------------------------------------------ the assistant
class Assistant:
    def __init__(self, service: StyleService, api_key: str | None = None):
        self.api_key = api_key or load_api_key()
        self.tools = make_tools(service)
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def reply(self, messages: list[dict]) -> dict:
        """`messages`: the conversation, [{"role": "user"|"assistant", "content": str}, ...], ending with the user."""
        if not self.api_key:
            raise AssistantUnavailable("The assistant needs GEMINI_API_KEY (environment or .env).")
        from google import genai
        from google.genai import types
        self._client = self._client or genai.Client(api_key=self.api_key)

        contents = [types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
                    for m in messages]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, tools=self.tools, temperature=0.3,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=MAX_TOOL_CALLS))

        last_error = None
        for model in MODELS:
            for attempt in range(2):
                try:
                    response = self._client.models.generate_content(model=model, contents=contents, config=config)
                    return {"reply": (response.text or "").strip() or "Sorry, I couldn't find an answer to that.",
                            "tools_used": self._tool_calls(response), "model": model}
                except Exception as error:
                    last_error, code = error, getattr(error, "code", None)
                    if not (code == 429 or (isinstance(code, int) and code >= 500)):
                        raise AssistantUnavailable(f"Gemini error: {error}") from None
                    time.sleep(2 * (attempt + 1))              # busy: retry, then fall back to the next model
        raise AssistantUnavailable(f"Gemini is busy, please try again in a moment ({last_error}).")

    @staticmethod
    def _tool_calls(response) -> list[dict]:
        calls = []
        for content in response.automatic_function_calling_history or []:
            for part in content.parts or []:
                if part.function_call is not None:
                    calls.append({"name": part.function_call.name, "args": dict(part.function_call.args or {})})
        return calls
