"""
mock_leaf_server.py
=====================
Stand-in for Module 01 (Leaf AI) while your teammate is still developing.
Returns realistic fake data in the EXACT schema colab_leaf_server.py will
eventually return, so app_ui.py can be built and tested end-to-end today.

Run locally:  python mock_leaf_server.py
Then in config.py:  MODULE_01_LEAF_URL = "http://127.0.0.1:8001"

Swap this URL for the real ngrok one the moment your teammate has it —
nothing else in app_ui.py needs to change, because the schema is identical.
"""
import random

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from typing import List

app = FastAPI(title="MOCK Module 01 - Leaf AI")


@app.get("/health")
def health():
    return {"status": "ok", "module": "leaf-ai-MOCK"}


@app.post("/analyze-leaves")
async def analyze_leaves(
    tree_id: str = Form(...),
    x_api_key: str = Form(...),
    images: List[UploadFile] = File(...),
):
    # Simulate real inference latency so your UI's loading states get tested too
    n = len(images)
    return {
        "tree_id": tree_id,
        "lsi": round(random.uniform(20, 80), 1),
        "leaf_disease_summary": {
            "Leaf Spot": {
                "pct_leaves_affected": round(random.uniform(10, 50), 1),
                "avg_severity": round(random.uniform(0.2, 0.8), 2),
            },
            "Leaf Rust": {
                "pct_leaves_affected": round(random.uniform(5, 30), 1),
                "avg_severity": round(random.uniform(0.1, 0.5), 2),
            },
        },
        "num_leaves_processed": n,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
