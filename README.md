# Convert PyG (<2.0) OPENABC data to a format readable by PyG >=2.0.

Environment assumptions:
- `openabc_old`: can read original data `.pt/.pt.zip` (PyG <2.0).
- `pyg_default` (your target new env): PyG >=2.0.

## 0) Prepare legacy data
1. Download the legacy OPENABC2 dataset from the [OpenABC Dataset Release Page](https://zenodo.org/records/6399454#.YkTglzwpA5k).
2. Unzip the dataset archive into a folder, e.g., `OPENABC2_DATASET`.
   The folder structure should look like:

```
OPENABC2_DATASET/
  processed/
    xxxx.pt.zip
    ...
```

## 1) Dump intermediate dicts (old env)
Run in `openabc_old`. Outputs neutral `.raw.pt` files (no PyG classes) to a writable folder.

```bash
# Activate the old environment (openabc_old) then run:
python datagen/utilities/dump_intermediate_pyg.py \
  --src OPENABC2_DATASET/processed \
  --dst OPENABC2_DATASET/processed_raw
```

This converts all `.pt/.pt.zip` files in `OPENABC2_DATASET/processed` to `.raw.pt` files in `OPENABC2_DATASET/processed_raw`, readable via `torch.load(path, weights_only=False)` in the new env.

They are large; next we rebuild zipped PyG files.

## 2) Rebuild new-format PyG files (new env)
Run in `pyg_default` (or your target new env). Takes `.raw.pt` and writes new-format `.pt.zip` (or plain `.pt` if you drop `--zip`).

```bash
# Activate the new environment (pyg_default) then run:
python datagen/utilities/rebuild_pyg_from_intermediate.py \
  --src OPENABC2_DATASET/processed_raw \
  --dst OPENABC2_DATASET/processed_new \
  --zip  # keep zipped layout; drop this flag to write plain .pt
```

## 3) Verify consistency via manifests
Generate hashes in each env, then compare. Use the same `--max` in both runs (set `0` for all files).

```bash
# openabc_old
python datagen/utilities/make_manifest.py \
  --src OPENABC2_DATASET/processed \
  --out OPENABC2_DATASET/manifest_old.csv \
  --max 10  # set 0 for full scan
```

```bash
# pyg_default (new env)
python datagen/utilities/make_manifest.py \
  --src OPENABC2_DATASET/processed_new \
  --out OPENABC2_DATASET/manifest_new.csv \
  --max 10  # match the old manifest
```

```bash
# pyg_default (new env)
python datagen/utilities/compare_manifest.py \
  --old OPENABC2_DATASET/manifest_old.csv \
  --new OPENABC2_DATASET/manifest_new.csv
```

We expect no differences when the conversion is successful.

The final output should be like:
> No dtype/shape/hash/num_nodes mismatches on common keys.

### Notes
- Manifests hash tensors (`sha256` of bytes) and record `dtype/shape/num_nodes`; mismatches will be listed.
- If you need a specific file checked, re-run `make_manifest.py --max 0` and `compare_manifest.py` for full coverage.
- The conversion keeps graph semantics intact (`edge_index`, `node_type`, `synVec`, `synID/stepID` etc.), only storage format changes.
