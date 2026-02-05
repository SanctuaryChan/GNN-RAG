import json
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Merge eval results with input prompts using 'id', and filter low-hit samples.")
    parser.add_argument("--pred", required=True, help="Path to predictions.jsonl (contains 'id' and 'input')")
    parser.add_argument("--eval", required=True, help="Path to detailed_eval_result.jsonl (contains 'id', 'hit', etc.)")
    parser.add_argument("--output", required=True, help="Output file path for filtered samples")
    parser.add_argument("--hit-threshold", type=int, default=0, help="Keep samples with hit <= this value (default: 0)")
    args = parser.parse_args()

    # Step 1: Load predictions (with input)
    pred_map = load_predictions(args.pred)

    # Step 2: Process eval file and merge
    kept_count = 0
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
                    merged = {
                        "id": sample_id,
                        "input": pred_data.get("input", ""),  # 或者可能是 "question", "prompt" 等字段
                        "prediction": eval_data.get("prediction", []),
                        "ground_truth": eval_data.get("ground_truth", []),
                        "hit": hit,
                        "f1": eval_data.get("f1"),
                        "acc": eval_data.get("acc"),
                        "precission": eval_data.get("precission"),  # 注意拼写是 precision（但你的数据里是 precission）
                        "recall": eval_data.get("recall")
                    }
                    fout.write(json.dumps(merged, ensure_ascii=False) + '\n')
                    kept_count += 1
            except Exception as e:
                print(f"Error processing line: {line[:100]}... ({e})")

    print(f"Done. Kept {kept_count} samples with hit <= {args.hit_threshold}. Saved to {args.output}")

if __name__ == "__main__":
    main()