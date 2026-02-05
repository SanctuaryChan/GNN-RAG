#!/usr/bin/env python3
import argparse
import json
import os
import re
import string
from typing import Iterable, List, Tuple


def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def split_prediction(pred) -> List[str]:
    if pred is None:
        return []
    if isinstance(pred, list):
        return pred
    return str(pred).split("\n")


def eval_metrics(pred_lines: List[str], answers: List[str]) -> Tuple[float, float, float, int, int, List[str], List[str]]:
    pred_str = " ".join(pred_lines)
    matched_answers = [a for a in answers if match(pred_str, a)]
    missing_answers = [a for a in answers if not match(pred_str, a)]
    matched = len(matched_answers)

    precision = matched / len(pred_lines) if pred_lines else 0.0
    recall = matched / len(answers) if answers else 0.0
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    hit = 1 if matched > 0 else 0
    hit1 = 0
    if pred_lines:
        for a in answers:
            if match(pred_lines[0], a):
                hit1 = 1
                break
    return f1, precision, recall, hit, hit1, matched_answers, missing_answers


def extract_reasoning_paths(input_text: str) -> List[str]:
    if not input_text:
        return []
    marker = "Reasoning Paths:"
    idx = input_text.find(marker)
    if idx == -1:
        return []
    seg = input_text[idx + len(marker):]
    for cut in ["\nQuestion:", "\nChoices:", "\n[/INST]", "Question:"]:
        pos = seg.find(cut)
        if pos != -1:
            seg = seg[:pos]
            break
    lines = [l.strip() for l in seg.splitlines() if l.strip()]
    return [l for l in lines if "->" in l]


def candidates_from_paths(paths: List[str]) -> List[str]:
    cands: List[str] = []
    for p in paths:
        parts = [x.strip() for x in p.split("->") if x.strip()]
        if parts:
            cands.append(parts[-1])
    return cands


def line_matches_any(line: str, items: List[str]) -> bool:
    for it in items:
        if match(line, it):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="predictions.jsonl path")
    parser.add_argument(
        "--output",
        default=None,
        help="output jsonl (default: same dir as predictions, metric_drop_cases.jsonl)",
    )
    parser.add_argument("--min-f1", type=float, default=1.0, help="only output cases with f1 < min-f1")
    parser.add_argument("--max-output", type=int, default=0, help="max cases to output (0 = all)")
    parser.add_argument(
        "--include-paths",
        action="store_true",
        help="include reasoning paths in output (can be large)",
    )
    args = parser.parse_args()

    pred_path = args.predictions
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)

    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(pred_path)), "metric_drop_cases.jsonl")

    total = 0
    emitted = 0
    with open(out_path, "w") as fout:
        for line_no, row in enumerate(iter_jsonl(pred_path), start=1):
            total += 1
            qid = row.get("id")
            question = row.get("question")
            prediction = row.get("prediction")
            answers = row.get("ground_truth") or []
            input_text = row.get("input")

            pred_lines = split_prediction(prediction)
            f1, precision, recall, hit, hit1, matched_answers, missing_answers = eval_metrics(pred_lines, answers)

            if f1 >= args.min_f1:
                continue

            paths = extract_reasoning_paths(input_text or "")
            cand_answers = candidates_from_paths(paths) if paths else []

            empty_lines: List[str] = []
            answer_lines: List[str] = []
            extra_lines: List[str] = []
            extra_supported: List[str] = []
            extra_unsupported: List[str] = []

            for l in pred_lines:
                if not str(l).strip():
                    empty_lines.append(l)
                    continue
                has_gt = line_matches_any(l, answers)
                if has_gt:
                    answer_lines.append(l)
                    continue
                extra_lines.append(l)
                if cand_answers:
                    if line_matches_any(l, cand_answers):
                        extra_supported.append(l)
                    else:
                        extra_unsupported.append(l)

            record = {
                "line_no": line_no,
                "id": qid,
                "question": question,
                "hit": hit,
                "hit1": hit1,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "ground_truth": answers,
                "matched_answers": matched_answers,
                "missing_answers": missing_answers,
                "pred_lines": pred_lines,
                "answer_lines": answer_lines,
                "extra_lines": extra_lines,
                "empty_lines": empty_lines,
            }

            if cand_answers:
                record["cand_answers"] = cand_answers
                record["extra_supported_by_paths"] = extra_supported
                record["extra_unsupported_by_paths"] = extra_unsupported

            if args.include_paths:
                record["reasoning_paths"] = paths

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            emitted += 1
            if args.max_output > 0 and emitted >= args.max_output:
                break

    print(
        json.dumps(
            {
                "total": total,
                "emitted": emitted,
                "output": out_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
