import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter JSONL lines where 'hit' == 0.")
    parser.add_argument("input", help="Path to the input .jsonl file")
    parser.add_argument("output", help="Path to the output .jsonl file")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("hit") == 0:
                    fout.write(line + '\n')
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {line[:100]}... ({e})")

    print(f"Filtered samples with hit=0 written to {args.output}")

if __name__ == "__main__":
    main()