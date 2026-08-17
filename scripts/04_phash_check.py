"""
Step 4 — near-duplicate / same-tree leakage check.

Runs on CPU, needs no GPU, and can run while something else trains.

If your 952 images are ~950 different trees, this finds almost nothing and you
proceed with the random split. If they are ~60 trees at 15 views each, it finds
large clusters, and any image-level random split leaks the same trunk into both
train and test — which makes the reported test accuracy a memorisation score.

The output tells you which of the three cases you are in, and writes
outputs/qc/duplicate_clusters.csv listing every cluster that spans splits.

Run:  python scripts/04_phash_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, PROJECT_ROOT                      # noqa: E402

import imagehash                                                      # noqa: E402
from PIL import Image                                                 # noqa: E402


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main() -> None:
    cfg = load_config()
    proc = cfg.path("paths", "processed_root")
    qc = cfg.path("paths", "qc_root")
    qc.mkdir(parents=True, exist_ok=True)

    manifest = proc / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found — run scripts/02_build_masks.py first")

    df = pd.read_csv(manifest).reset_index(drop=True)
    hs = int(cfg["phash"]["hash_size"])
    thr = int(cfg["phash"]["hamming_threshold"])

    print(f"hashing {len(df)} images (hash_size={hs}) ...")
    hashes = []
    for p in df.image_path:
        with Image.open(PROJECT_ROOT / p) as im:
            hashes.append(imagehash.phash(im.convert("RGB"), hash_size=hs))

    print(f"comparing pairs (threshold = {thr}) ...")
    dsu = DSU(len(df))
    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            d = hashes[i] - hashes[j]
            if d <= thr:
                dsu.union(i, j)
                pairs.append((i, j, d))

    df["cluster"] = [dsu.find(i) for i in range(len(df))]
    sizes = df.cluster.value_counts()
    multi = sizes[sizes > 1]

    print(f"\nnear-duplicate pairs: {len(pairs)}")
    print(f"clusters with >1 image: {len(multi)}")
    print(f"images inside such clusters: {int(multi.sum())} / {len(df)} "
          f"({100 * multi.sum() / len(df):.1f}%)")
    if len(multi):
        print(f"largest cluster: {int(multi.max())} images")

    # which clusters straddle a split boundary -> actual leakage
    spanning = (df[df.cluster.isin(multi.index)]
                .groupby("cluster")
                .filter(lambda g: g.split.nunique() > 1))

    if len(spanning):
        out = qc / "duplicate_clusters.csv"
        spanning.sort_values(["cluster", "split"])[
            ["cluster", "split", "class_name", "file_name"]
        ].to_csv(out, index=False)
        print(f"\n!! {spanning.cluster.nunique()} cluster(s) span more than one "
              f"split — {len(spanning)} images are leaking.")
        print(f"   listed in {out}")
        print("   Fix cheaply: move every member of a cluster into whichever "
              "split holds the majority of it.")
    else:
        print("\nNo cluster spans a split. Random split is safe on this "
              "evidence — record that in your methods section.")

    df[["split", "file_name", "class_name", "cluster"]].to_csv(
        qc / "phash_clusters_all.csv", index=False)
    print(f"\nfull cluster assignment: {qc / 'phash_clusters_all.csv'}")


if __name__ == "__main__":
    main()
