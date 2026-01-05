"""
Generate a manifest (CSV) summarizing PyG Data files without requiring PyG compatibility
across environments. Run this in the environment that can read the target files
(e.g., old env for original data, new env for converted data).

Each row contains:
  stem, key, dtype, shape, sha256, num_nodes (if present)

Usage:
  python datagen/utilities/make_manifest.py --src <folder> --out manifest_old.csv --max 100
"""
import argparse
import glob
import hashlib
import io
import os
import os.path as osp
import zipfile

import torch
import pandas as pd


def choose_file(stem: str, folder: str):
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
        stem = osp.splitext(osp.basename(zp))[0]  # drops .zip, keeps .pt
        if stem.endswith(".pt"):
            stem = stem[:-3]
        stems.add(stem)
    for pt in glob.glob(osp.join(folder, "*.pt")):
        stem = osp.splitext(osp.basename(pt))[0]
        if stem not in stems and not stem.startswith("pre_"):
            stems.add(stem)
    return sorted(stems)


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


def tensor_hash(t: torch.Tensor):
    # Move to CPU, ensure contiguous bytes, hash with sha256
    b = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def make_manifest(src: str, out: str, max_files: int):
    stems = list_stems(src)
    if max_files:
        stems = stems[:max_files]
    rows = []
    for i, stem in enumerate(stems, 1):
        path = choose_file(stem, src)
        if path is None:
            continue
        data = load_any(path)
        # num_nodes may not be in keys
        num_nodes = getattr(data, "num_nodes", None)
        keys = data.keys if not callable(data.keys) else data.keys()
        for k in keys:
            v = getattr(data, k)
            if isinstance(v, torch.Tensor):
                rows.append(
                    {
                        "stem": stem,
                        "key": k,
                        "dtype": str(v.dtype),
                        "shape": str(tuple(v.shape)),
                        "sha256": tensor_hash(v),
                        "num_nodes": num_nodes,
                    }
                )
            else:
                rows.append(
                    {
                        "stem": stem,
                        "key": k,
                        "dtype": type(v).__name__,
                        "shape": "",
                        "sha256": hashlib.sha256(str(v).encode()).hexdigest(),
                        "num_nodes": num_nodes,
                    }
                )
        if i % 50 == 0 or i == len(stems):
            print(f"Processed {i}/{len(stems)} files")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Wrote manifest to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create a hash manifest for PyG Data files.")
    ap.add_argument("--src", required=True, help="Folder containing .pt/.pt.zip")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--max", type=int, default=0, help="Max files to process (0 = all)")
    args = ap.parse_args()
    make_manifest(args.src, args.out, args.max)
