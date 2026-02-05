#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from typing import Iterable

from datasets import Dataset, DatasetDict, load_from_disk


def stable_keep(sample_id: str, mod: int, keep: int) -> bool:
    h = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(h, 16) % mod == keep


def pick_ids(ds: Dataset, mod: int, keep: int, id_field: str) -> Dataset:
    def _f(example):
        sid = example.get(id_field, "")
        if sid is None:
            sid = ""
        sid = str(sid)
        if sid == "":
            return False
        return stable_keep(sid, mod, keep)

    return ds.filter(_f)


def ensure_id_field(ds: Dataset) -> str:
    if "id" in ds.column_names:
        return "id"
    if "question" in ds.column_names:
        return "question"
    raise ValueError("Dataset must contain 'id' or 'question' field for stable sampling.")


def parse_splits(arg: str) -> Iterable[str]:
    return [s.strip() for s in arg.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="Create a deterministic sample dataset (by id hash).")
    parser.add_argument("--data_path", required=True, help="Root path containing dataset directory")
    parser.add_argument("--dataset", required=True, help="Dataset name under data_path")
    parser.add_argument("--out_name", default=None, help="Output dataset dir name (default: <dataset>_sXX)")
    parser.add_argument("--mod", type=int, default=10, help="Hash modulo (default: 10 for 10%%)")
    parser.add_argument("--keep", type=int, default=0, help="Keep bucket (default: 0)")
    parser.add_argument("--splits", default="test", help="Comma-separated splits to sample (default: test)")
    parser.add_argument("--save_ids", action="store_true", help="Save sampled ids to <out_dir>/sample_ids.json")
    args = parser.parse_args()

    in_dir = os.path.join(args.data_path, args.dataset)
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input dataset not found: {in_dir}")

    ds = load_from_disk(in_dir)

    if isinstance(ds, DatasetDict):
        split_names = parse_splits(args.splits)
        out_dd = DatasetDict()
        for split in split_names:
            if split not in ds:
                raise ValueError(f"Split '{split}' not found in dataset. Available: {list(ds.keys())}")
            split_ds = ds[split]
            id_field = ensure_id_field(split_ds)
            sampled = pick_ids(split_ds, args.mod, args.keep, id_field)
            out_dd[split] = sampled
            print(f"Split {split}: {len(sampled)} / {len(split_ds)} ({len(sampled)/max(1,len(split_ds))*100:.2f}%)")
        out_ds = out_dd
    else:
        id_field = ensure_id_field(ds)
        out_ds = pick_ids(ds, args.mod, args.keep, id_field)
        print(f"Split (single): {len(out_ds)} / {len(ds)} ({len(out_ds)/max(1,len(ds))*100:.2f}%)")

    if args.out_name is None:
        pct = int(round(100 / args.mod)) if args.mod != 0 else 0
        out_name = f"{args.dataset}_s{pct}"
    else:
        out_name = args.out_name

    out_dir = os.path.join(args.data_path, out_name)
    if os.path.exists(out_dir):
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    out_ds.save_to_disk(out_dir)

    if args.save_ids:
        ids = []
        if isinstance(out_ds, DatasetDict):
            for split, split_ds in out_ds.items():
                id_field = ensure_id_field(split_ds)
                ids.extend([str(x) for x in split_ds[id_field]])
        else:
            id_field = ensure_id_field(out_ds)
            ids = [str(x) for x in out_ds[id_field]]
        with open(os.path.join(out_dir, "sample_ids.json"), "w") as f:
            json.dump(ids, f, indent=2)

    print(f"Saved sampled dataset to: {out_dir}")


if __name__ == "__main__":
    main()
