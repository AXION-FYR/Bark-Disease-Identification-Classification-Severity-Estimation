"""
mock_decision_server.py
=========================
Stand-in for Module 03 (Decision Engine) while your teammate is still
developing. Returns realistic fake recommendations matching the EXACT
schema colab_decision_server.py will eventually return.

Run locally:  python mock_decision_server.py
Then in config.py:  MODULE_03_DECISION_URL = "http://127.0.0.1:8003"

Swap this URL for the real ngrok one once your teammate has it wired.
"""
import random

import uvicorn
from fastapi import FastAPI, Header
from pydantic import BaseModel
from typing import Dict, Optional, List

app = FastAPI(title="MOCK Module 03 - Decision Engine")


class DiseaseSeverity(BaseModel):
    pct_bark: float
    stage: str
    qsi: Optional[float] = None
    pct_leaves_affected: Optional[float] = None
    avg_severity: Optional[float] = None


class BarkResult(BaseModel):
    per_disease_bark: Dict[str, DiseaseSeverity]
    num_views_processed: Optional[int] = None
    bsi: Optional[float] = None
    primary_bark_disease: Optional[str] = None
    circumferential_spread_pct: Optional[float] = None
    girdling_risk: Optional[str] = None


class LeafResult(BaseModel):
    lsi: float
    leaf_disease_summary: Dict[str, DiseaseSeverity]
    num_leaves_processed: int


class WeatherForecast(BaseModel):
    location: str
    next_7_days: List[dict]


class MarketPrices(BaseModel):
    cinnamon_grade: str
    price_per_kg_lkr: float
    trend: str


class DecisionRequest(BaseModel):
    tree_id: str
    leaf: LeafResult
    bark: BarkResult
    weather_forecast: WeatherForecast
    market_prices: MarketPrices


@app.get("/health")
def health():
    return {"status": "ok", "module": "decision-engine-MOCK"}


@app.post("/generate-recommendation")
def generate_recommendation(req: DecisionRequest, x_api_key: str = Header(...)):
    worst_disease = max(
        req.bark.per_disease_bark.items(), key=lambda kv: kv[1].pct_bark, default=(None, None)
    )[0]
    risk = random.choice(["Low", "Medium", "High"])
    return {
        "tree_id": req.tree_id,
        "overall_risk": risk,
        "treatment_plan": [
            {"action": f"Apply targeted treatment for {worst_disease}", "priority": 1, "timing": "within 3 days"},
            {"action": "Monitor circumferential spread weekly", "priority": 2, "timing": "ongoing"},
        ],
        "harvest_recommendation": {
            "recommended": risk != "High",
            "reason": f"Overall risk assessed as {risk} based on bark and leaf indices",
            "optimal_window": "next 2-3 weeks" if risk != "High" else "delay until treated",
        },
        "summary": f"[MOCK] Tree {req.tree_id} shows {worst_disease or 'no significant disease'}; risk level {risk}.",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)
