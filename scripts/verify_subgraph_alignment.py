#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json_or_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        # Peek first non-space char
        first = ""
        while True:
            ch = f.read(1)
            if ch == "":
                break
            if not ch.isspace():
                first = ch
                break
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for item in data:
                if isinstance(item, dict):
                    yield item
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _load_id_map(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _load_name_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {str(k): str(v) for k, v in data.items()}


def _pick_key(sample: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in sample:
            v = sample[k]
            if isinstance(v, (str, int)):
                return str(v)
    return None


def _normalize_token(x: Any,
                     id_list: Optional[List[str]],
                     name_map: Optional[Dict[str, str]] = None) -> str:
    if isinstance(x, dict):
        # Prefer 'text' or 'kb_id'
        if "text" in x:
            return str(x["text"])
        if "kb_id" in x:
            return str(x["kb_id"])
        return str(x)
    if isinstance(x, int):
        if id_list is not None and 0 <= x < len(id_list):
            x = id_list[x]
        else:
            x = str(x)
    # Some ids may be numeric strings
    if isinstance(x, str) and x.isdigit():
        idx = int(x)
        if id_list is not None and 0 <= idx < len(id_list):
            x = id_list[idx]
    if name_map is not None:
        key = str(x)
        if key in name_map:
            return name_map[key]
    return str(x)


def _extract_triples(sample: Dict[str, Any],
                     entity_list: Optional[List[str]],
                     rel_list: Optional[List[str]],
                     entity_name_map: Optional[Dict[str, str]] = None) -> List[Tuple[str, str, str]]:
    triples = None
    if "subgraph" in sample and isinstance(sample["subgraph"], dict) and "tuples" in sample["subgraph"]:
        triples = sample["subgraph"]["tuples"]
    elif "graph" in sample:
        triples = sample["graph"]

    if triples is None:
        return []

    normed: List[Tuple[str, str, str]] = []
    for t in triples:
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            continue
        h, r, tail = t
        h_n = _normalize_token(h, entity_list, entity_name_map)
        r_n = _normalize_token(r, rel_list, None)
        t_n = _normalize_token(tail, entity_list, entity_name_map)
        normed.append((h_n, r_n, t_n))
    return normed


def _build_index(samples: Iterable[Dict[str, Any]], align_by: str) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for i, s in enumerate(samples):
        key = None
        if align_by == "id":
            key = _pick_key(s, ["id", "qid", "question_id"])
        elif align_by == "question":
            q = s.get("question")
            if isinstance(q, str):
                key = q.strip().lower()
        elif align_by == "index":
            key = str(i)
        if key is None:
            continue
        if key in idx:
            # keep first occurrence
            continue
        idx[key] = s
    return idx


def _load_rog_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.rog_json:
        return list(_read_json_or_jsonl(args.rog_json))

    # --rog-hf can be a hub name or a local datasets.load_from_disk path
    rog_source = args.rog_hf
    if rog_source is None:
        return []

    try:
        from datasets import load_dataset, load_from_disk
    except Exception as e:
        print("datasets library is required for --rog-hf. Error:", e, file=sys.stderr)
        raise

    if os.path.exists(rog_source):
        ds = load_from_disk(rog_source)
        if isinstance(ds, dict):
            if args.rog_split not in ds:
                raise KeyError(
                    f"Split '{args.rog_split}' not found in {rog_source} (available: {list(ds.keys())})"
                )
            ds = ds[args.rog_split]
        return [ds[i] for i in range(len(ds))]

    # Note: this may trigger network download if not cached.
    ds = load_dataset(rog_source, split=args.rog_split)
    return [ds[i] for i in range(len(ds))]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify whether RoG and non-RoG subgraphs are aligned (same triples) per sample.")
    parser.add_argument("--gnn-json", required=True, help="Path to GNN dataset json/jsonl (e.g., gnn/data/webqsp/test.json)")

    rog = parser.add_mutually_exclusive_group(required=True)
    rog.add_argument("--rog-json", help="Path to RoG dataset json/jsonl (local export)")
    rog.add_argument("--rog-hf", help="HuggingFace dataset name OR local datasets.load_from_disk path")
    parser.add_argument("--rog-split", default="test", help="Split name for HF dataset (default: test)")

    parser.add_argument("--align-by", choices=["id", "question", "index"], default="id",
                        help="How to align samples between datasets")
    parser.add_argument("--gnn-entities", help="Path to GNN entities.txt (for ID->string mapping)")
    parser.add_argument("--gnn-relations", help="Path to GNN relations.txt (for ID->string mapping)")
    parser.add_argument("--entity-name-map", help="Path to entities_names.json (ID->name mapping)")
    parser.add_argument("--report", type=int, default=5, help="Report up to N mismatched samples")
    parser.add_argument("--show-diff", action="store_true", help="Print sample-level missing triples for reported cases")
    parser.add_argument("--debug-key", help="Print normalized triples for a specific aligned key")
    parser.add_argument("--debug-n", type=int, default=10, help="How many triples to show in debug output")
    parser.add_argument("--diff-stats", action="store_true", help="Aggregate diff statistics across aligned samples")
    parser.add_argument("--topk", type=int, default=10, help="Top-K relations/entities to report for diff stats")
    parser.add_argument("--diff-entity", action="store_true", help="Also report top entities in diffs")

    args = parser.parse_args()

    ent_list = _load_id_map(args.gnn_entities)
    rel_list = _load_id_map(args.gnn_relations)
    name_map = _load_name_map(args.entity_name_map)

    # Load datasets
    gnn_samples = list(_read_json_or_jsonl(args.gnn_json))

    try:
        rog_samples = _load_rog_samples(args)
    except Exception:
        return 2

    gnn_idx = _build_index(gnn_samples, args.align_by)
    rog_idx = _build_index(rog_samples, args.align_by)

    common_keys = sorted(set(gnn_idx.keys()) & set(rog_idx.keys()))
    if not common_keys:
        print("No aligned samples found. Try --align-by question or index.")
        return 1

    if args.debug_key:
        key = args.debug_key
        if key not in gnn_idx or key not in rog_idx:
            print(f"debug-key '{key}' not found in both datasets (align_by={args.align_by}).")
            return 1
        gnn_triples = set(_extract_triples(gnn_idx[key], ent_list, rel_list, name_map))
        rog_triples = set(_extract_triples(rog_idx[key], None, None, None))
        inter = list(gnn_triples & rog_triples)
        only_g = list(gnn_triples - rog_triples)
        only_r = list(rog_triples - gnn_triples)
        print(f"Debug key={key}")
        print(f"GNN triples: {len(gnn_triples)} | RoG triples: {len(rog_triples)} | Intersection: {len(inter)}")
        print(f"Sample GNN triples (up to {args.debug_n}): {only_g[:args.debug_n] if only_g else list(gnn_triples)[:args.debug_n]}")
        print(f"Sample RoG triples (up to {args.debug_n}): {only_r[:args.debug_n] if only_r else list(rog_triples)[:args.debug_n]}")
        print(f"Sample intersection (up to {args.debug_n}): {inter[:args.debug_n]}")
        return 0

    total = 0
    exact = 0
    jaccard_sum = 0.0
    mismatches: List[Tuple[str, float, int, int]] = []
    total_gnn_triples = 0
    total_rog_triples = 0
    total_inter = 0
    total_only_g = 0
    total_only_r = 0
    rel_only_g: Counter[str] = Counter()
    rel_only_r: Counter[str] = Counter()
    ent_only_g: Counter[str] = Counter()
    ent_only_r: Counter[str] = Counter()

    for k in common_keys:
        gnn_triples = set(_extract_triples(gnn_idx[k], ent_list, rel_list, name_map))
        rog_triples = set(_extract_triples(rog_idx[k], None, None, None))
        if not gnn_triples and not rog_triples:
            continue
        total += 1
        inter_set = gnn_triples & rog_triples
        inter = len(inter_set)
        union = len(gnn_triples) + len(rog_triples) - inter
        jaccard = inter / union if union else 1.0
        jaccard_sum += jaccard
        if jaccard == 1.0:
            exact += 1
        else:
            mismatches.append((k, jaccard, len(gnn_triples), len(rog_triples)))
        if args.diff_stats:
            total_gnn_triples += len(gnn_triples)
            total_rog_triples += len(rog_triples)
            total_inter += inter
            only_g = gnn_triples - rog_triples
            only_r = rog_triples - gnn_triples
            total_only_g += len(only_g)
            total_only_r += len(only_r)
            for h, r, t in only_g:
                rel_only_g[r] += 1
                if args.diff_entity:
                    ent_only_g[h] += 1
                    ent_only_g[t] += 1
            for h, r, t in only_r:
                rel_only_r[r] += 1
                if args.diff_entity:
                    ent_only_r[h] += 1
                    ent_only_r[t] += 1

    if total == 0:
        print("No comparable samples with non-empty subgraphs.")
        return 1

    avg_jaccard = jaccard_sum / total
    print(f"Aligned samples compared: {total}")
    print(f"Exact match: {exact} ({exact/total:.2%})")
    print(f"Average Jaccard: {avg_jaccard:.4f}")
    if args.diff_stats:
        print("\nDiff stats (aggregated):")
        if total_gnn_triples > 0:
            print(f"- Total GNN triples: {total_gnn_triples}")
            print(f"- Total RoG triples: {total_rog_triples}")
            print(f"- Intersection: {total_inter} ({total_inter/total_gnn_triples:.2%} of GNN)")
            print(f"- Only in GNN: {total_only_g} ({total_only_g/total_gnn_triples:.2%} of GNN)")
        if total_rog_triples > 0:
            print(f"- Only in RoG: {total_only_r} ({total_only_r/total_rog_triples:.2%} of RoG)")
        if rel_only_g:
            print(f"\nTop relations only in GNN (top {args.topk}):")
            for r, c in rel_only_g.most_common(args.topk):
                print(f"  {r}: {c}")
        if rel_only_r:
            print(f"\nTop relations only in RoG (top {args.topk}):")
            for r, c in rel_only_r.most_common(args.topk):
                print(f"  {r}: {c}")
        if args.diff_entity:
            if ent_only_g:
                print(f"\nTop entities only in GNN diffs (top {args.topk}):")
                for e, c in ent_only_g.most_common(args.topk):
                    print(f"  {e}: {c}")
            if ent_only_r:
                print(f"\nTop entities only in RoG diffs (top {args.topk}):")
                for e, c in ent_only_r.most_common(args.topk):
                    print(f"  {e}: {c}")

    if mismatches:
        print(f"\nMismatched samples (showing up to {args.report}):")
        for k, j, gsz, rsz in mismatches[: args.report]:
            print(f"- key={k} | jaccard={j:.4f} | gnn={gsz} rog={rsz}")
            if args.show_diff:
                gnn_triples = set(_extract_triples(gnn_idx[k], ent_list, rel_list, name_map))
                rog_triples = set(_extract_triples(rog_idx[k], None, None, None))
                only_g = list(gnn_triples - rog_triples)[:10]
                only_r = list(rog_triples - gnn_triples)[:10]
                print(f"  only in GNN (up to 10): {only_g}")
                print(f"  only in RoG (up to 10): {only_r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
