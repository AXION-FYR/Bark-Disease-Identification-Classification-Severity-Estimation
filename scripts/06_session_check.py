"""
Step 6 — capture-session leakage check, from filenames alone.

Roboflow filenames keep the original stem:  "sc (271)_JPG.rf.<hash>.JPG"
The prefix ('sc') and the number (271) survive. Consecutive numbers were almost
always shot in the same session, often of the same trunk from different angles.

If a run of consecutive numbers straddles the train/test boundary, an
image-level random split has leaked the same tree into both — and the reported
test accuracy is partly a memorisation score.

This runs in a second, needs no processed cache, and is a sharper signal than
perceptual hashing for datasets with sequential filenames.

Run:  python scripts/06_session_check.py
      python scripts/06_session_check.py --gap 2   # merge runs across gaps of 2
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                                    # noqa: E402
from src.coco_utils import find_annotation_file, load_coco, image_class  # noqa: E402

# "sc (271)_JPG.rf.abc123.JPG" -> ('sc', 271)
STEM_RE = re.compile(r"^\s*([A-Za-z]+)\s*\((\d+)\)")


def parse_stem(file_name: str) -> tuple[str, int] | None:
    m = STEM_RE.match(file_name)
    if not m:
        return None
    return m.group(1).lower(), int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=1,
                    help="numbers within this gap join the same session run")
    args = ap.parse_args()

    cfg = load_config()
    raw_root = cfg.path("paths", "raw_root")

    # file_name -> (split, class, prefix, number)
    items = []
    unparsed = []
    for split in cfg["splits"] + ["."]:
        folder = raw_root / split if split != "." else raw_root
        ann = find_annotation_file(folder)
        if ann is None:
            continue
        for rec in load_coco(ann):
            p = parse_stem(rec.file_name)
            cls, _ = image_class(rec)
            if p is None:
                unparsed.append(rec.file_name)
                continue
            items.append((split, cls, p[0], p[1], rec.file_name))

    if not items:
        sys.exit("could not parse any filenames — is the naming scheme different?")

    print(f"parsed {len(items)} filenames; {len(unparsed)} unparsed")
    if unparsed:
        print(f"  examples: {unparsed[:3]}")

    # group by prefix, sort by number, cut into runs of consecutive numbers
    by_prefix: dict[str, list] = defaultdict(list)
    for split, cls, pre, num, fn in items:
        by_prefix[pre].append((num, split, cls, fn))

    runs = []
    for pre, lst in by_prefix.items():
        lst.sort()
        cur = [lst[0]]
        for entry in lst[1:]:
            if entry[0] - cur[-1][0] <= args.gap:
                cur.append(entry)
            else:
                runs.append((pre, cur))
                cur = [entry]
        runs.append((pre, cur))

    print(f"\nfilename prefixes: {dict((p, len(v)) for p, v in by_prefix.items())}")
    print(f"consecutive runs (gap<={args.gap}): {len(runs)}")

    # --- the question that matters: runs spanning train and test
    def splits_in(run):
        return {e[1] for e in run}

    train_test = [(p, r) for p, r in runs if {"train", "test"} <= splits_in(r)]
    any_span = [(p, r) for p, r in runs if len(splits_in(r)) > 1]

    print(f"runs spanning >1 split      : {len(any_span)}")
    print(f"runs spanning TRAIN and TEST: {len(train_test)}")

    n_imgs_tt = sum(len(r) for _, r in train_test)
    print(f"images inside train/test-spanning runs: {n_imgs_tt} / {len(items)} "
          f"({100 * n_imgs_tt / len(items):.1f}%)")

    if train_test:
        print("\nboundary crossings (the pairs to worry about):")
        shown = 0
        for pre, run in train_test:
            for a, b in zip(run, run[1:]):
                if a[1] != b[1] and {a[1], b[1]} == {"train", "test"}:
                    print(f"  {pre} ({a[0]}) [{a[1]}/{a[2]}]  <->  "
                          f"{pre} ({b[0]}) [{b[1]}/{b[2]}]")
                    shown += 1
                    if shown >= 15:
                        break
            if shown >= 15:
                break

    qc = cfg.path("paths", "qc_root")
    qc.mkdir(parents=True, exist_ok=True)
    out = qc / "session_runs.csv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("run_id,prefix,number,split,class_name,spans_splits,file_name\n")
        for i, (pre, run) in enumerate(runs):
            spans = int(len(splits_in(run)) > 1)
            for num, split, cls, fn in run:
                fh.write(f"{i},{pre},{num},{split},{cls},{spans},{fn}\n")
    print(f"\nfull run assignment: {out}")

    print("\nHow to read this:")
    print("  0 train/test-spanning runs  -> sessions are already separated by "
          "split. Say so in your methods; it is a real strength.")
    print("  A few                       -> move the minority side of each run "
          "into the other split. Cheap, and mostly closes the gap.")
    print("  Many                        -> do not re-split this week. Report "
          "both the full test score and a score on the non-spanning subset, "
          "and state the gap honestly.")


if __name__ == "__main__":
    main()
