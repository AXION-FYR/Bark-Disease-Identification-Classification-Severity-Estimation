# Axion Integration — Architecture & Run Order

## Architecture: Hub-and-Spoke over ngrok tunnels

```
                    ┌─────────────────────────────────────┐
                    │         YOUR LOCAL MACHINE           │
                    │                                       │
  User ──uploads──▶ │  app_ui.py (Streamlit)                │
                    │     │                                 │
                    │     ├─▶ tree_severity_bridge.py        │
                    │     │   (Module 02 — in-process,       │
                    │     │    same machine, no HTTP)        │
                    │     │                                 │
                    │     ├──HTTP POST──▶ ngrok URL 1 ───────┼──▶ Colab: Module 01 (Leaf AI)
                    │     │                                 │
                    │     └──HTTP POST──▶ ngrok URL 2 ───────┼──▶ Colab: Module 03 (Decision Engine)
                    └─────────────────────────────────────┘
```

**Why this is the fastest option:** you don't touch the internals of Module 01 or
Module 03 at all. You only add a few cells to the *bottom* of each Colab
notebook that expose the existing inference function as a REST endpoint. Your
local machine becomes the orchestrator since it already has Module 02 and the
UI together — that removes one network hop entirely.

**No CORS setup needed.** Streamlit's server-side Python (not the browser)
makes the `requests.post()` calls to Colab, so this is server-to-server
traffic, not a cross-origin browser request.

## What each file is

| File | Runs where | Purpose |
|---|---|---|
| `colab_leaf_server.py` | Paste into last cells of **Module 01's** Colab notebook | Wraps existing leaf inference in a FastAPI `/analyze-leaves` endpoint, exposes it via ngrok |
| `colab_decision_server.py` | Paste into last cells of **Module 03's** Colab notebook | Wraps existing decision engine in a FastAPI `/generate-recommendation` endpoint, exposes it via ngrok |
| `tree_severity_bridge.py` | Your local machine | Adapts your existing `tree_severity.py` 5-stage pipeline to accept in-memory uploaded images and return a clean dict — no disk I/O |
| `app_ui.py` | Your local machine (`streamlit run app_ui.py`) | The orchestrator UI: upload widgets, calls all 3 modules, renders the report |
| `config.py` | Your local machine | One place to paste today's ngrok URLs (they change on restart on the free tier) |

## 2-Day Run Order

**Day 1 — morning:** Each teammate independently pastes their server wrapper
into their own Colab notebook and confirms it responds via `curl` or the
FastAPI `/docs` Swagger page. This can happen in parallel — nobody blocks
anybody.

**Day 1 — afternoon:** You wire `tree_severity_bridge.py` into your actual
pipeline (swap the `TODO` markers for your real function names/classes), and
smoke-test it locally with a folder of sample images before touching
Streamlit at all.

**Day 1 — evening:** Build `app_ui.py`, point it at mock/dummy ngrok URLs or
even a local mock server first, confirm the *flow* works end-to-end.

**Day 2 — morning:** Swap in real ngrok URLs from teammates, do a full
end-to-end run with real images, fix schema mismatches (there will be a
few — this is normal).

**Day 2 — afternoon:** Polish the UI, error states, and demo script. Keep
both Colab runtimes alive and connected for the entire demo — free-tier
Colab disconnects after ~90 min idle, and ngrok free URLs die if the
notebook cell stops running.

## Locked JSON Schemas

**Module 02 (Bark AI) → Module 03, the `bark` field.** The only field
Module 03 actually *requires* is `per_disease_bark` — a disease-name to
percentage/stage map:

```json
{
  "per_disease_bark": {
    "stripecanker": {"pct_bark": 34.0, "stage": "Early control"},
    "Rough bark":   {"pct_bark": 56.0, "stage": "Active management"}
  },
  "num_views_processed": 15
}
```

`stage` is one of `"Preventive" | "Early control" | "Active management" |
"Severe outbreak"`, same wording your Bark AI pipeline already produces —
don't relabel it on the way in. `bsi`, `primary_bark_disease`,
`circumferential_spread_pct`, and `girdling_risk` are also sent alongside
this as optional context (Module 03 doesn't need them, but they're free
and might be useful for logging/debugging).

**Module 01 (Leaf AI) response, the `leaf` field:**

```json
{
  "lsi": 62.4,
  "leaf_disease_summary": {
    "Leaf Spot": {"pct_leaves_affected": 40.0, "avg_severity": 0.55}
  },
  "num_leaves_processed": 15
}
```

**Full request to Module 03** (`POST /generate-recommendation`) combines
both of the above plus weather and market data — see
`colab_decision_server.py`'s `DecisionRequest` model for the exact
Pydantic contract; that file is the source of truth if this README and
the code ever drift.

## Known friction points (plan for these now)

1. **ngrok URLs are ephemeral on the free tier.** Every time a teammate
   restarts their Colab runtime, they get a new URL. Put the URL in
   `config.py` (or better, a shared Google Doc/Slack) and update it each
   session. If anyone has a free ngrok account, a *reserved* static domain
   removes this problem entirely (`ngrok config add-authtoken`, then use
   `ngrok http --domain=your-name.ngrok-free.app 8000`).
2. **Colab idle timeout.** Keep the tab open, or use a "keep alive" trick
   during the demo window.
3. **Multipart image uploads over HTTP.** The leaf server takes 15 files —
   watch total payload size; compress/resize before upload if it's slow on
   your teammate's connection.
4. **Schema drift.** Lock the JSON schemas below *now* between all 3 of you
   so nobody's endpoint silently changes field names the night before
   the demo.
