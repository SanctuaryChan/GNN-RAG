import json
import argparse
import re
import os

def load_predictions(pred_file):
    """从 predictions.jsonl 加载 id -> {input, ...} 的映射"""
    pred_map = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pred_map[data["id"]] = data
            except Exception as e:
                print(f"Warning: Skipping invalid line in {pred_file}: {line[:100]}... ({e})")
    return pred_map

QUESTION_PATTERN = re.compile(r"Question:\s*(.*?)\s*\[/INST\]", re.S)

def extract_question_from_input(input_text):
    """从 input prompt 中提取 Question 文本。"""
    if not input_text:
        return ""
    match = QUESTION_PATTERN.search(input_text)
    if match:
        return match.group(1).strip()
    idx = input_text.rfind("Question:")
    if idx != -1:
        return input_text[idx + len("Question:"):].strip()
    return ""

def normalize_question(question):
    """标准化问题文本以便匹配 test.info。"""
    return " ".join(question.strip().split())

def load_test_info(test_info_file):
    """从 test.info 加载 question -> cand 的映射"""
    cand_map = {}
    total = 0
    dup = 0
    with open(test_info_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                question = normalize_question(data.get("question", ""))
                if not question:
                    continue
                total += 1
                cand = data.get("cand")
                if question in cand_map:
                    dup += 1
                    # 若已有为空，则用当前 cand 补充
                    if not cand_map[question] and cand:
                        cand_map[question] = cand
                    continue
                cand_map[question] = cand
            except Exception as e:
                print(f"Warning: Skipping invalid line in {test_info_file}: {line[:100]}... ({e})")
    return cand_map, total, dup

def resolve_entities_names_path(user_path):
    """Resolve entities_names.json path from user input or common locations."""
    if user_path:
        return user_path
    candidates = [
        os.path.join(os.getcwd(), "entities_names.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "entities_names.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def load_entities_names(path):
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load entities_names.json from {path}: {e}")
        return None

def map_cand_to_names(cand, entities_names):
    if not cand or not entities_names:
        return None
    mapped = []
    for item in cand:
        if isinstance(item, list) and len(item) >= 2:
            mid = item[0]
            score = item[1]
            name = entities_names.get(mid, mid)
            mapped.append([name, score])
        else:
            mapped.append(item)
    return mapped

def main():
    parser = argparse.ArgumentParser(description="Merge eval results with input prompts using 'id', and filter low-hit samples.")
    parser.add_argument("--data-dir", help="Directory containing predictions.jsonl and detailed_eval_result.jsonl")
    parser.add_argument("--pred", help="Path to predictions.jsonl (contains 'id' and 'input')")
    parser.add_argument("--eval", help="Path to detailed_eval_result.jsonl (contains 'id', 'hit', etc.)")
    parser.add_argument("--pred-name", help="Filename under --data-dir for predictions.jsonl")
    parser.add_argument("--eval-name", help="Filename under --data-dir for detailed_eval_result.jsonl")
    parser.add_argument("--output", required=True, help="Output file path for filtered samples")
    parser.add_argument("--hit-threshold", type=int, default=0, help="Keep samples with hit <= this value (default: 0)")
    parser.add_argument("--test-info", help="Path to test.info jsonl (merge 'cand' by question)")
    parser.add_argument("--entities-names", help="Path to entities_names.json for cand MID -> name mapping")
    parser.add_argument("--cand-name", action="store_true", help="Replace cand MID with name (keeps original in cand_mid)")
    args = parser.parse_args()

    # If data-dir provided, compose pred/eval paths when missing
    if args.data_dir:
        if not args.pred:
            pred_name = args.pred_name or "predictions.jsonl"
            pred_path = os.path.join(args.data_dir, pred_name)
            if args.pred_name is None and not os.path.exists(pred_path):
                alt = os.path.join(args.data_dir, "prediction.jsonl")
                if os.path.exists(alt):
                    pred_path = alt
            args.pred = pred_path
        if not args.eval:
            eval_name = args.eval_name or "detailed_eval_result.jsonl"
            args.eval = os.path.join(args.data_dir, eval_name)

    if not args.pred or not args.eval:
        parser.error("--pred and --eval are required unless --data-dir is provided")

    # Step 1: Load predictions (with input)
    pred_map = load_predictions(args.pred)

    # Step 1.5: Load test.info (optional)
    cand_map = {}
    test_info_total = 0
    test_info_dup = 0
    if args.test_info:
        cand_map, test_info_total, test_info_dup = load_test_info(args.test_info)

    # Step 1.6: Load entities_names.json (optional)
    entities_names_path = resolve_entities_names_path(args.entities_names)
    entities_names = load_entities_names(entities_names_path)

    # Step 2: Process eval file and merge
    kept_count = 0
    cand_matched = 0
    with open(args.eval, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                eval_data = json.loads(line)
                hit = eval_data.get("hit")
                sample_id = eval_data.get("id")

                if hit is not None and hit <= args.hit_threshold:
                    # Merge with prediction data
                    pred_data = pred_map.get(sample_id, {})
                    question = pred_data.get("question") or extract_question_from_input(pred_data.get("input", ""))
                    question_norm = normalize_question(question) if question else ""
                    cand = cand_map.get(question_norm) if question_norm and cand_map else None

                    merged = {
                        "id": sample_id,
                        "input": pred_data.get("input", ""),  # 或者可能是 "question", "prompt" 等字段
                        "question": question,
                        "prediction": eval_data.get("prediction", []),
                        "ground_truth": eval_data.get("ground_truth", []),
                        "hit": hit,
                        "f1": eval_data.get("f1"),
                        "acc": eval_data.get("acc"),
                        "precission": eval_data.get("precission"),  # 注意拼写是 precision（但你的数据里是 precission）
                        "recall": eval_data.get("recall")
                    }
                    if args.test_info:
                        merged["cand"] = cand
                        if entities_names is not None and cand is not None:
                            cand_named = map_cand_to_names(cand, entities_names)
                            if args.cand_name:
                                merged["cand_mid"] = cand
                                merged["cand"] = cand_named
                            else:
                                merged["cand_named"] = cand_named
                        if cand is not None:
                            cand_matched += 1
                    fout.write(json.dumps(merged, ensure_ascii=False) + '\n')
                    kept_count += 1
            except Exception as e:
                print(f"Error processing line: {line[:100]}... ({e})")

    print(f"Done. Kept {kept_count} samples with hit <= {args.hit_threshold}. Saved to {args.output}")
    if args.test_info:
        print(f"test.info loaded: {test_info_total} questions, {test_info_dup} duplicates")
        print(f"cand matched for kept samples: {cand_matched}/{kept_count}")
    if entities_names_path:
        print(f"entities_names.json: {entities_names_path}")

if __name__ == "__main__":
    main()
