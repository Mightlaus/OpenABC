"""
Rebuild PyG Data files from neutral intermediate dicts.

Run this in the new environment (e.g., conda env `pyg_default`).
Reads *.raw.pt produced by dump_intermediate_pyg.py and writes
new-format PyG Data objects (.pt or .pt.zip).
"""
import argparse
import glob
import os
import os.path as osp
import zipfile
import sys
import types

import torch
from torch_geometric.data import Data

# Shim for dummy storage class that may be present in intermediate pickles.
# This matches the _DummyStorage injected in dump_intermediate_pyg.py.
if "_DummyStorage" not in globals():
    class _DummyStorage(dict):
        pass
    sys.modules[__name__]._DummyStorage = _DummyStorage
if "torch_geometric.data.storage" not in sys.modules:
    storage_mod = types.ModuleType("torch_geometric.data.storage")
    storage_mod.NodeStorage = _DummyStorage
    storage_mod.EdgeStorage = _DummyStorage
    storage_mod.GlobalStorage = _DummyStorage
    sys.modules["torch_geometric.data.storage"] = storage_mod


def collect_files(src: str):
    return glob.glob(osp.join(src, "*.raw.pt"))


def rebuild_data(d: dict):
    data = Data(**{k: v for k, v in d.items() if k != "num_nodes"})
    if "num_nodes" in d:
        data.num_nodes = d["num_nodes"]
    return data


def save_data(data: Data, out_path: str, zip_out: bool):
    if zip_out:
        pt_path = osp.splitext(out_path)[0] if out_path.endswith(".zip") else out_path
        torch.save(data, pt_path)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pt_path, arcname=osp.basename(pt_path))
        os.remove(pt_path)
    else:
        torch.save(data, out_path)


def main(args):
    os.makedirs(args.dst, exist_ok=True)
    files = collect_files(args.src)
    total = len(files)
    print(f"Found {total} intermediate files in {args.src}")
    for i, fp in enumerate(files, 1):
        d = torch.load(fp, weights_only=False)
        data = rebuild_data(d)
        base = osp.basename(fp).replace(".raw.pt", ".pt")
        if args.zip:
            out_path = osp.join(args.dst, f"{base}.zip")
        else:
            out_path = osp.join(args.dst, base)
        save_data(data, out_path, zip_out=args.zip)
        if i % 200 == 0 or i == total:
            print(f"Rebuilt {i}/{total}")
    print(f"Done. New-format files in {args.dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild PyG Data from intermediate dicts.")
    parser.add_argument("--src", required=True, help="Folder with *.raw.pt produced by dump_intermediate_pyg.py")
    parser.add_argument("--dst", required=True, help="Folder to write new-format PyG Data files")
    parser.add_argument("--zip", action="store_true", help="Write .pt.zip (if omitted, write plain .pt)")
    args = parser.parse_args()
    main(args)
