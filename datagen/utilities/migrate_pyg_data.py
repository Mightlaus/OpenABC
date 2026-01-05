import argparse
import glob
import os
import os.path as osp
import zipfile
import io
import sys
import types

import torch
from torch_geometric.data import Data

# Some older/newer pickles may reference torch_geometric.data.storage.* which
# is absent in PyG 1.7.0. Provide a lightweight shim to let unpickling succeed.
if "torch_geometric.data.storage" not in sys.modules:
    storage_mod = types.ModuleType("torch_geometric.data.storage")
    class _DummyStorage(dict):
        pass
    storage_mod.NodeStorage = _DummyStorage
    storage_mod.EdgeStorage = _DummyStorage
    storage_mod.GlobalStorage = _DummyStorage
    sys.modules["torch_geometric.data.storage"] = storage_mod


def load_old(path: str):
    """Load a Data object from either .pt or .pt.zip (old PyG format)."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            inner = osp.splitext(osp.basename(path))[0]
            raw = zf.read(inner)
            buffer = io.BytesIO(raw)
            try:
                return torch.load(buffer, weights_only=False)
            except TypeError:
                buffer.seek(0)
                return torch.load(buffer)
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def save_new(data: Data, out_path: str, zip_out: bool):
    """Save a fresh Data instance so it matches the current PyG format."""
    # Re-instantiate to rebuild the internal store layout.
    new_data = Data(**{k: getattr(data, k) for k in data.keys})
    if hasattr(data, "num_nodes"):
        new_data.num_nodes = data.num_nodes

    if zip_out:
        # If caller wants zipped output, save to .pt then zip it.
        pt_path = osp.splitext(out_path)[0] if out_path.endswith(".zip") else out_path
        torch.save(new_data, pt_path)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pt_path, arcname=osp.basename(pt_path))
        os.remove(pt_path)
    else:
        torch.save(new_data, out_path)


def collect_files(src: str):
    """Prefer .pt.zip if both zipped and plain exist for the same stem."""
    zip_files = glob.glob(osp.join(src, "*.pt.zip"))
    zip_stems = {osp.basename(f)[:-4] for f in zip_files}  # remove .zip
    pt_files = []
    for f in glob.glob(osp.join(src, "*.pt")):
        base = osp.basename(f)
        if base in zip_stems:
            continue
        if base.startswith("pre_"):  # skip PyG cache helpers
            continue
        pt_files.append(f)
    return zip_files + pt_files


def main(args):
    os.makedirs(args.dst, exist_ok=True)
    files = collect_files(args.src)
    total = len(files)
    print(f"Found {total} files to migrate from {args.src} -> {args.dst}")
    for i, fp in enumerate(files, 1):
        data = load_old(fp)
        out_name = osp.basename(fp)
        out_path = osp.join(args.dst, out_name)
        save_new(data, out_path, zip_out=args.zip)
        if i % 200 == 0 or i == total:
            print(f"Migrated {i}/{total}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate old PyG Data files to new format.")
    parser.add_argument("--src", required=True, help="Source folder containing old .pt or .pt.zip files")
    parser.add_argument("--dst", required=True, help="Destination folder for migrated files")
    parser.add_argument("--zip", action="store_true", help="Write zipped output (.pt.zip)")
    main(parser.parse_args())
