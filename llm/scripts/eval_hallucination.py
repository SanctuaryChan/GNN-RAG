#!/usr/bin/env python3
import argparse
import json
import os
import re
import string
from typing import Dict, Iterable, List, Optional, Set, Tuple

# 通过判断预测是否落在 GNN 候选集合中，评估幻觉相关指标（可过滤解释性输出）。
# 主要指标：
# - EC（Evidence Consistency）：预测是否落在候选集合内
# - HR（Hallucination Rate）：1 - EC
# - SH（Strict Hallucination）：是否出现“任何一个”候选集合外预测


def normalize(text: str) -> str:
    # 与 evaluate_results.py 的 normalize 规则保持一致：
    # 小写 + 去标点 + 去冠词 + 去多余空白。
    text = text.lower()
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\b(<pad>)\b", " ", text)
    return " ".join(text.split())


def load_entities_names(path: str) -> Dict[str, str]:
    # 将 mid -> 实体名的映射载入，用于把 GNN 候选实体转成可读文本。
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[dict]:
    # 逐行读取 jsonl，忽略空行。
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def default_test_json(gnn_info_path: str) -> str:
    # 默认与 test.info 同目录下的 test.json。
    return os.path.join(os.path.dirname(gnn_info_path), "test.json")


def load_gnn_candidates(
    test_json_path: str, gnn_info_path: str, entities_names: Dict[str, str]
) -> Dict[str, List[str]]:
    # 按 test.json 与 test.info 的行序对齐（与 predict_answer.py 一致）。
    # 目的是构造 qid -> 候选实体名列表 的映射。
    id_to_cands: Dict[str, List[str]] = {}
    with open(test_json_path) as f_json, open(gnn_info_path) as f_info:
        for line_json, line_info in zip(f_json, f_info):
            data_json = json.loads(line_json)
            data_info = json.loads(line_info)
            qid = data_json.get("id")
            if qid is None:
                continue
            cands = []
            for cand in data_info.get("cand", []):
                if not cand:
                    continue
                mid = cand[0]
                name = entities_names.get(mid, mid)
                cands.append(name)
            id_to_cands[qid] = cands
    return id_to_cands


def strip_leading_marker(line: str) -> str:
    # 去掉开头的项目符号/编号（如 "1. ", "- "）。
    return re.sub(r"^\s*[\-\*\d\.\)\:]+\s+", "", line).strip()


def split_segments(line: str) -> List[str]:
    # 按常见分隔符切分答案，减少解释性噪声。
    # 对短句尝试再按 "and" 拆分。
    parts = re.split(r"[;,]", line)
    segments: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if " and " in part and len(part) < 80:
            segments.extend([p.strip() for p in part.split(" and ") if p.strip()])
        else:
            segments.append(part)
    return segments


def extract_predictions(
    prediction: str,
    cand_norm: Set[str],
    max_line_len: int,
    explain_keywords: List[str],
    containment: bool,
    keep_only_cand: bool,
    drop_explain_unless_cand: bool,
) -> Tuple[List[str], bool]:
    # 允许解释输出的情况下提取候选答案行。
    # 逻辑：
    # 1) 先按行拆分，再去掉编号/项目符号；
    # 2) 对包含解释词的长行直接丢弃；
    # 3) 对每行按分隔符切成短片段，作为候选实体行。
    lines = [strip_leading_marker(l) for l in prediction.split("\n") if l.strip()]
    explain_hit = False
    kept: List[str] = []
    def seg_matches_cand(seg_norm: str) -> bool:
        if not seg_norm:
            return False
        if seg_norm in cand_norm:
            return True
        if containment and any(c in seg_norm for c in cand_norm):
            return True
        return False

    for line in lines:
        lower_line = line.lower()
        explain_line = any(kw in lower_line for kw in explain_keywords)
        if explain_line:
            explain_hit = True
            if drop_explain_unless_cand:
                line_norm = normalize(line)
                if not seg_matches_cand(line_norm):
                    continue
            elif len(line) > max_line_len:
                continue
        for seg in split_segments(line):
            if not seg:
                continue
            seg_norm = normalize(seg)
            match = seg_matches_cand(seg_norm)
            if keep_only_cand and not match:
                continue
            if len(seg) > max_line_len and not match:
                continue
            kept.append(seg)
    return kept, explain_hit


def build_norm_set(items: List[str]) -> Set[str]:
    # 将候选实体或预测实体归一化成集合。
    return {normalize(i) for i in items if normalize(i)}


def compute_metrics(pred_norm: Set[str], cand_norm: Set[str]) -> Tuple[float, float, int]:
    # EC/HR/SH 定义见 docs/design_graph_constrained_decoding.md。
    # 空预测视为 0（不计幻觉，但可统计空回答率）。
    if not pred_norm:
        return 0.0, 0.0, 0
    inter = pred_norm.intersection(cand_norm)
    ec = len(inter) / max(1, len(pred_norm))
    hr = 1.0 - ec
    sh = 1 if len(pred_norm - cand_norm) > 0 else 0
    return ec, hr, sh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="predictions.jsonl path")
    parser.add_argument("--gnn-info", required=True, help="test.info path")
    parser.add_argument("--test-json", default=None, help="test.json path (default: sibling of test.info)")
    parser.add_argument(
        "--entities-names",
        default="../entities_names.json",
        help="entities_names.json path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="详细评估文件路径（默认：与 predictions.jsonl 同目录的 hallucination_eval_detail.jsonl）",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="汇总结果文件路径（默认：与 predictions.jsonl 同目录的 hallucination_eval_result.jsonl）",
    )
    parser.add_argument("--max-line-len", type=int, default=80)
    parser.add_argument(
        "--explain-keywords",
        default="because,therefore,according,as a result,based on,possible answers,the answer",
    )
    parser.add_argument("--containment", action="store_true")
    parser.add_argument(
        "--keep-only-cand",
        action="store_true",
        help="只保留命中候选实体的片段",
    )
    parser.add_argument(
        "--drop-explain-unless-cand",
        action="store_true",
        help="解释性行默认丢弃，除非该行包含候选实体",
    )
    parser.add_argument("--max-cand", type=int, default=0, help="truncate cand list (0 = keep all)")
    parser.add_argument(
        "--log-samples",
        type=int,
        default=0,
        help="打印前 N 条样本的抽取/匹配日志（0 表示关闭）",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="日志输出路径（默认：stdout）",
    )
    args = parser.parse_args()

    # 解析 test.json 路径用于 id 对齐。
    test_json_path = args.test_json or default_test_json(args.gnn_info)
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"test.json not found: {test_json_path}")

    entities_names = load_entities_names(args.entities_names)
    id_to_cands = load_gnn_candidates(test_json_path, args.gnn_info, entities_names)

    # 默认输出到 predictions.jsonl 同目录，文件名固定为 hallucination_eval_detail.jsonl
    pred_dir = os.path.dirname(os.path.abspath(args.predictions))
    out_path = args.output or os.path.join(pred_dir, "hallucination_eval_detail.jsonl")
    summary_path = args.summary_output or os.path.join(pred_dir, "hallucination_eval_result.jsonl")
    explain_keywords = [k.strip().lower() for k in args.explain_keywords.split(",") if k.strip()]

    total = 0
    missing = 0
    ec_list: List[float] = []
    hr_list: List[float] = []
    sh_list: List[int] = []
    empty_list: List[int] = []
    explain_list: List[int] = []

    log_out = open(args.log_path, "w") if args.log_path else None

    def log(msg: str) -> None:
        if log_out:
            log_out.write(msg + "\n")
        else:
            print(msg)

    # 输出逐样本评估结果，便于调试与回溯。
    with open(out_path, "w") as fout:
        for row in iter_jsonl(args.predictions):
            total += 1
            qid = row.get("id")
            if qid not in id_to_cands:
                missing += 1
                continue
            prediction = row.get("prediction", "") or ""
            cands = id_to_cands[qid]
            if args.max_cand > 0:
                cands = cands[: args.max_cand]
            cand_norm = build_norm_set(cands)

            pred_lines, explain_hit = extract_predictions(
                prediction,
                cand_norm=cand_norm,
                max_line_len=args.max_line_len,
                explain_keywords=explain_keywords,
                containment=args.containment,
                keep_only_cand=args.keep_only_cand,
                drop_explain_unless_cand=args.drop_explain_unless_cand,
            )
            pred_norm = build_norm_set(pred_lines)

            ec, hr, sh = compute_metrics(pred_norm, cand_norm)
            empty = 1 if not pred_norm else 0

            ec_list.append(ec)
            hr_list.append(hr)
            sh_list.append(sh)
            empty_list.append(empty)
            explain_list.append(1 if explain_hit else 0)

            if args.log_samples > 0 and total <= args.log_samples:
                log(
                    json.dumps(
                        {
                            "id": qid,
                            "prediction_raw": prediction,
                            "pred_lines": pred_lines,
                            "pred_norm": sorted(list(pred_norm)),
                            "cand_norm_sample": sorted(list(cand_norm))[:10],
                            "ec": ec,
                            "hr": hr,
                            "sh": sh,
                            "empty": empty,
                            "explain": 1 if explain_hit else 0,
                        },
                        ensure_ascii=False,
                    )
                )

            fout.write(
                json.dumps(
                    {
                        "id": qid,
                        "pred_lines": pred_lines,
                        "pred_norm": sorted(list(pred_norm)),
                        "cand_norm": sorted(list(cand_norm)),
                        "ec": ec,
                        "hr": hr,
                        "sh": sh,
                        "empty": empty,
                        "explain": 1 if explain_hit else 0,
                    }
                )
                + "\n"
            )

    def mean(xs: List[float]) -> float:
        return sum(xs) / max(1, len(xs))

    summary = {
        "samples": total,
        "missing_cand": missing,
        "ec": mean(ec_list),
        "hr": mean(hr_list),
        "sh": mean([float(x) for x in sh_list]),
        "empty_rate": mean([float(x) for x in empty_list]),
        "explain_rate": mean([float(x) for x in explain_list]),
    }
    print(json.dumps(summary, indent=2))
    with open(summary_path, "w") as fsum:
        fsum.write(json.dumps(summary) + "\n")

    if log_out:
        log_out.close()


if __name__ == "__main__":
    main()
