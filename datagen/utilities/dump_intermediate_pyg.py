"""
Dump PyG Data objects (old format) to a neutral intermediate dict.

Run this inside the old environment (e.g., conda env `openabc_old`)
so it can read the legacy .pt/.pt.zip files. The output files are
pure Python/tensor dicts (no PyG classes), so they can be loaded
by any newer PyTorch/PyG.
"""
import argparse
import glob
import io
import os
import os.path as osp
import zipfile
import sys
import types

import torch

# Shim for old pickles that reference torch_geometric.data.storage.*
if "torch_geometric.data.storage" not in sys.modules:
    storage_mod = types.ModuleType("torch_geometric.data.storage")
    class _DummyStorage(dict):
        pass
    storage_mod.NodeStorage = _DummyStorage
    storage_mod.EdgeStorage = _DummyStorage
    storage_mod.GlobalStorage = _DummyStorage
    sys.modules["torch_geometric.data.storage"] = storage_mod


def collect_files(src: str):
    """Prefer .pt.zip if both exist; skip pre_* cache files."""
    zip_files = glob.glob(osp.join(src, "*.pt.zip"))
    zip_stems = {osp.basename(f)[:-4] for f in zip_files}  # drop .zip
    pt_files = []
    for f in glob.glob(osp.join(src, "*.pt")):
        base = osp.basename(f)
        if base in zip_stems or base.startswith("pre_"):
            continue
        pt_files.append(f)
    return zip_files + pt_files


def load_pyg(fp: str):
    """Load old-format PyG Data from .pt or .pt.zip."""
    if fp.endswith(".zip"):
        with zipfile.ZipFile(fp) as zf:
            inner = osp.splitext(osp.basename(fp))[0]
            raw = zf.read(inner)
            return torch.load(io.BytesIO(raw))
    return torch.load(fp)


def to_intermediate_dict(data):
    """Convert a PyG Data object to a plain dict of tensors/metadata."""
    out = {}
    for k in data.keys:
        out[k] = getattr(data, k)
    # Keep num_nodes if present
    if hasattr(data, "num_nodes"):
        out["num_nodes"] = data.num_nodes
    return out


def main(args):
    os.makedirs(args.dst, exist_ok=True)
    files = collect_files(args.src)
    total = len(files)
    print(f"Found {total} files in {args.src}")
    for i, fp in enumerate(files, 1):
        data = load_pyg(fp)
        out_dict = to_intermediate_dict(data)
        base = osp.splitext(osp.basename(fp))[0]  # strip .zip if present
        if base.endswith(".pt"):
            base = base[:-3]
        out_path = osp.join(args.dst, f"{base}.raw.pt")
        torch.save(out_dict, out_path)
        if i % 200 == 0 or i == total:
            print(f"Dumped {i}/{total}")
    print(f"Done. Intermediate files in {args.dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump old PyG Data to neutral dict files.")
    parser.add_argument("--src", required=True, help="Folder containing legacy .pt/.pt.zip files")
    parser.add_argument("--dst", required=True, help="Folder to write .raw.pt intermediate files")
    args = parser.parse_args()
    main(args)
