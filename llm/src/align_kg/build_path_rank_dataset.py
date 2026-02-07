import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
import json
import random
import glob
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

import utils


def random_walk_path(graph, start_nodes, hop):
    if not start_nodes:
        return []
    node = random.choice(start_nodes)
    if node not in graph:
        return []
    path = []
    length = random.randint(1, hop)
    for _ in range(length):
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            break
        nxt = random.choice(neighbors)
        rel = graph[node][nxt]['relation']
        path.append((node, rel, nxt))
        node = nxt
    return path


def build_samples(sample, hop, n_neg, use_verbalizer):
    question = sample['question']
    q_entity = sample['q_entity']
    a_entity = sample['a_entity']
    graph = utils.build_graph(sample['graph'])

    positives = []
    pos_paths = utils.get_truth_paths(q_entity, a_entity, graph)
    for p in pos_paths:
        if not p:
            continue
        rel_path = [r for _, r, _ in p]
        answer = p[-1][-1]
        path_text = utils.verbalize_path(p, question=question) if use_verbalizer else utils.path_to_string(p)
        positives.append({
            "question": question,
            "path": p,
            "rel_path": rel_path,
            "answer": answer,
            "label": 1,
            "path_text": path_text,
        })

    negatives = []
    answer_set = set(a_entity)
    start_nodes = [e for e in q_entity if e in graph]
    seen = set()
    trials = 0
    while len(negatives) < n_neg and trials < n_neg * 10:
        trials += 1
        p = random_walk_path(graph, start_nodes, hop)
        if not p:
            continue
        end_ent = p[-1][-1]
        if end_ent in answer_set:
            continue
        sig = tuple((h, r, t) for h, r, t in p)
        if sig in seen:
            continue
        seen.add(sig)
        rel_path = [r for _, r, _ in p]
        path_text = utils.verbalize_path(p, question=question) if use_verbalizer else utils.path_to_string(p)
        negatives.append({
            "question": question,
            "path": p,
            "rel_path": rel_path,
            "answer": end_ent,
            "label": 0,
            "path_text": path_text,
        })

    return positives + negatives


def _load_any_dataset(path, split):
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_dict.json")):
        return load_from_disk(path)[split]
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, f"{split}.*")))
        if candidates:
            data_file = candidates[0]
            ext = os.path.splitext(data_file)[1].lstrip(".").lower()
            if ext in {"jsonl", "json"}:
                return load_dataset("json", data_files={split: data_file}, split=split)
            if ext == "arrow":
                return load_dataset("arrow", data_files={split: data_file}, split=split)
            if ext == "parquet":
                return load_dataset("parquet", data_files={split: data_file}, split=split)
    return load_dataset(path, split=split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument("-d", "--dataset", type=str, default="webqsp")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="datasets/PathRankData")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--n_neg", type=int, default=2)
    parser.add_argument("--use_verbalizer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entities_names_path", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    if args.entities_names_path:
        utils.set_entities_names_path(args.entities_names_path)

    input_file = os.path.join(args.data_path, args.dataset)
    output_dir = os.path.join(args.output_path, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.save_name == "":
        args.save_name = f"{args.dataset}_{args.split}.jsonl"

    dataset = _load_any_dataset(input_file, args.split)
    out_path = os.path.join(output_dir, args.save_name)

    with open(out_path, "w") as fout:
        for sample in tqdm(dataset, total=len(dataset)):
            records = build_samples(sample, args.hop, args.n_neg, args.use_verbalizer)
            for rec in records:
                fout.write(json.dumps(rec) + "\n")

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
