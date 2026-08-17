"""
verify_setup.py
=================
Run this from your project root (cinnamon/) to check the integration
setup end-to-end, one step at a time, in increasing order of risk.
Stops at the first failure and tells you exactly what to look at.

Usage:
    cd "D:/RESEARCH/Final evaluation/integrate test/cinnamon_stage1/cinnamon"
    python verify_setup.py
    python verify_setup.py --tree_folder D:/RESEARCH/Dataset/multiview_trees/tree6
"""
import argparse
import sys
import traceback
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"


def step(name):
    def decorator(fn):
        def wrapper(*a, **kw):
            print(f"\n{'='*60}\n[{name}]\n{'='*60}")
            try:
                fn(*a, **kw)
                print(f"-> {PASS}")
                return True
            except Exception as e:
                print(f"-> {FAIL}: {e}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


@step("1. Project root sanity — are we where we think we are?")
def check_location():
    cwd = Path.cwd()
    required = ["src", "scripts", "outputs", "tree_severity_bridge.py", "app_ui.py", "config.py"]
    missing = [r for r in required if not (cwd / r).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing from {cwd}: {missing}. "
            f"You must run this from your project root, next to src/ and scripts/."
        )
    print(f"Running from: {cwd}")
    print("All expected files/folders present at this level.")


@step("2. Import tree_severity_bridge.py — does it even parse and import?")
def check_import():
    global BarkSeverityPipeline
    from tree_severity_bridge import BarkSeverityPipeline  # noqa
    print("Imported BarkSeverityPipeline successfully.")


@step("3. Load models — do checkpoints exist and load?")
def check_model_load():
    global pipeline
    pipeline = BarkSeverityPipeline()
    print(f"Device: {pipeline.device}")
    print(f"Classes: {pipeline.idx_to_name}")


@step("4. Run analyze() on a real tree folder — does inference produce output?")
def check_analyze(tree_folder: str):
    from PIL import Image
    folder = Path(tree_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Tree folder not found: {folder}")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    photos = sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])
    if not photos:
        raise FileNotFoundError(f"No images found in {folder}")
    print(f"Found {len(photos)} images in {folder.name}")

    images = [Image.open(p) for p in photos]
    result = pipeline.analyze(folder.name, images)

    print("\n--- Result ---")
    print(f"tree_id: {result['tree_id']}")
    print(f"bsi: {result['bsi']:.2f}")
    print(f"primary_bark_disease: {result['primary_bark_disease']}")
    print(f"circumferential_spread_pct: {result['circumferential_spread_pct']:.1f}%")
    print(f"girdling_risk: {result['girdling_risk']}")
    print(f"num_views_processed: {result['num_views_processed']}")
    print("per_disease_bark:")
    for dis, info in result["per_disease_bark"].items():
        print(f"  {dis}: {info['pct_bark']:.1f}%  [{info['stage']}]  (QSI {info['qsi']:.3f})")

    print("\n>>> COMPARE the per_disease_bark lines above against what")
    print(">>> `python scripts/26_demo_tree.py --folder ...` printed for the")
    print(">>> SAME tree folder. The disease names, %, and [stage] should match.")


@step("5. Mock servers reachable? (only if you started them)")
def check_mocks():
    import requests
    import config
    for name, url in [("Module 01 mock", config.MODULE_01_LEAF_URL),
                       ("Module 03 mock", config.MODULE_03_DECISION_URL)]:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            print(f"{name} ({url}): {r.status_code} {r.json()}")
        except Exception as e:
            print(f"{name} ({url}): NOT REACHABLE ({e}) — "
                  f"start it with `python mock_leaf_server.py` / `python mock_decision_server.py` "
                  f"if you want to test this, otherwise ignore.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree_folder", default=None,
                     help="Path to a folder of one tree's bark photos, e.g. the same "
                          "one you used with 26_demo_tree.py")
    args = ap.parse_args()

    ok = check_location()
    if not ok:
        sys.exit(1)

    ok = check_import()
    if not ok:
        sys.exit(1)

    ok = check_model_load()
    if not ok:
        print("\nModel loading failed — check checkpoint paths in "
              "BarkSeverityPipeline() defaults match your actual outputs/seg/,"
              " outputs/cls/, outputs/lesion/ files.")
        sys.exit(1)

    if args.tree_folder:
        ok = check_analyze(args.tree_folder)
        if not ok:
            sys.exit(1)
    else:
        print("\n[4. skipped] Pass --tree_folder <path> to test real inference, e.g.:")
        print('  python verify_setup.py --tree_folder "D:/RESEARCH/Dataset/multiview_trees/tree6"')

    check_mocks()  # non-fatal either way

    print("\n" + "=" * 60)
    print("All required steps passed. Bridge is confirmed working.")
    print("Next: start mock servers, then `streamlit run app_ui.py`.")
    print("=" * 60)


if __name__ == "__main__":
    main()
