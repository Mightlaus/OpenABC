"""
Lightweight checker to verify converted PyG Data files match the originals.

Usage (run in any environment with access to both dirs):
    python datagen/utilities/verify_conversion.py \
        --orig OPENABC2_DATASET/processed \
        --new  OPENABC2_DATASET/processed_new \
        --max 50

It will:
 - Find common stems in both dirs (preferring .pt.zip, otherwise .pt).
 - Load original and converted Data.
 - Compare keys, tensor shapes/values, and scalar metadata.
 - Stop early on mismatches and report a summary.
"""
import argparse
import glob
import io
import os
import os.path as osp
import zipfile

import torch


def choose_file(stem: str, folder: str):
    """Pick .pt.zip if present, else .pt; return full path or None."""
    zip_path = osp.join(folder, f"{stem}.pt.zip")
    pt_path = osp.join(folder, f"{stem}.pt")
    if osp.exists(zip_path):
        return zip_path
    if osp.exists(pt_path):
        return pt_path
    return None


def list_stems(folder: str):
    stems = set()
    for zp in glob.glob(osp.join(folder, "*.pt.zip")):
        stem = osp.splitext(osp.basename(zp))[0]
        if stem.endswith(".pt"):
            stem = stem[:-3]
        stems.add(stem)
    for pt in glob.glob(osp.join(folder, "*.pt")):
        stem = osp.splitext(osp.basename(pt))[0]
        if stem not in stems:  # prefer zip if both exist
            stems.add(stem)
    return stems


def load_any(path: str):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            inner = osp.splitext(osp.basename(path))[0]
            raw = zf.read(inner)
            try:
                return torch.load(io.BytesIO(raw), weights_only=False)
            except TypeError:
                return torch.load(io.BytesIO(raw))
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def compare_data(a, b):
    keys_a = set(a.keys)
    keys_b = set(b.keys)
    if keys_a != keys_b:
        return False, f"key mismatch: {keys_a.symmetric_difference(keys_b)}"
    for k in keys_a:
        va, vb = getattr(a, k), getattr(b, k)
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if va.shape != vb.shape:
                return False, f"{k} shape {va.shape} != {vb.shape}"
            if not torch.equal(va, vb):
                return False, f"{k} values differ"
        else:
            if va != vb:
                return False, f"{k} differs: {va} vs {vb}"
    # num_nodes is stored outside keys
    if hasattr(a, "num_nodes") or hasattr(b, "num_nodes"):
        if getattr(a, "num_nodes", None) != getattr(b, "num_nodes", None):
            return False, f"num_nodes differ: {getattr(a,'num_nodes',None)} vs {getattr(b,'num_nodes',None)}"
    return True, "ok"


def main(args):
    stems_orig = list_stems(args.orig)
    stems_new = list_stems(args.new)
    common = sorted(list(stems_orig & stems_new))
    missing = sorted(list(stems_orig - stems_new))
    if missing:
        print(f"Warning: {len(missing)} stems missing in new folder. First few: {missing[:5]}")
    if args.max:
        common = common[: args.max]
    if not common:
        print("No common stems found.")
        return
    print(f"Comparing {len(common)} files...")
    for i, stem in enumerate(common, 1):
        orig_path = choose_file(stem, args.orig)
        new_path = choose_file(stem, args.new)
        a = load_any(orig_path)
        b = load_any(new_path)
        ok, msg = compare_data(a, b)
        if not ok:
            print(f"[FAIL] {stem}: {msg}")
            return
        if i % 50 == 0 or i == len(common):
            print(f"Checked {i}/{len(common)}")
    print("All checked files match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify converted PyG Data matches originals.")
    parser.add_argument("--orig", required=True, help="Folder with original .pt/.pt.zip")
    parser.add_argument("--new", required=True, help="Folder with converted .pt/.pt.zip")
    parser.add_argument("--max", type=int, default=100, help="Max files to compare (default 100)")
    args = parser.parse_args()
    main(args)
