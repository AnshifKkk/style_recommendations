"""Style Intelligence API.

    uvicorn backend.api:app --reload          # then open http://localhost:8000/docs

Endpoints
- GET /health                    service status
- GET /styles/top?limit=3        the top predicted styles for the next quarter
- GET /styles?limit=&offset=     the full ranking, with optional segment / product type filters
- GET /styles/{style_id}         product information, prediction, explanation and sales history of one style
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.model_service import StyleNotFound, StyleService, quarter_label
from backend.schemas import Health, StyleDetail, StyleList, TopStylesResponse


def create_app(service: StyleService | None = None) -> FastAPI:
    service = service or StyleService()
    app = FastAPI(
        title="Style Intelligence API",
        version="1.0.0",
        description=f"Predicted winning styles for {quarter_label(service.quarter)}, with explanations and sales history. "
                    "Only commercially relevant styles (top 10% by revenue last quarter) are ranked.",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
    if service.images_dir.is_dir():
        app.mount("/images", StaticFiles(directory=service.images_dir), name="images")

    @app.get("/health", response_model=Health, tags=["meta"])
    def health():
        return {"status": "ok", "quarter": service.quarter, "styles": service.n_styles}

    @app.get("/styles/top", response_model=TopStylesResponse, tags=["styles"])
    def top_styles(limit: int = Query(3, ge=1, le=50, description="Number of styles to return")):
        """The highest-ranked styles for the prediction quarter, with prediction score, category and sales history."""
        return service.top(limit)

    @app.get("/styles", response_model=StyleList, tags=["styles"])
    def list_styles(
        limit: int = Query(20, ge=1, le=200),
        offset: int = Query(0, ge=0),
        segment: str | None = Query(None, description="e.g. Ladieswear, Menswear, Divided, Sport, Baby/Children"),
        product_type: str | None = Query(None, description="e.g. Dress, Trousers, Sweater"),
        include_all: bool = Query(False, description="Include styles outside the business rule (they have no rank)"),
    ):
        """The ranking of styles, paginated and optionally filtered."""
        return service.list(limit, offset, segment, product_type, include_all)

    @app.get("/styles/{style_id}", response_model=StyleDetail, tags=["styles"],
             responses={404: {"description": "Unknown style_id"}})
    def style_detail(style_id: str):
        """Product information, prediction score, model explanation and historical performance of one style."""
        try:
            return service.get(style_id)
        except StyleNotFound:
            raise HTTPException(status_code=404, detail=f"Style '{style_id}' not found. "
                                                        "Use GET /styles to list valid style_ids.")

    return app


app = create_app()
