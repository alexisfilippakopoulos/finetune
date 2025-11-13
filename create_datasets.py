"""
sample_xsum.py

Usage examples:
  # choose 5% of the input file (e.g. 2000 -> 100)
  python create_dataset.py --input data/inference/amazon_train_2000.json --percent 5 --out_dir ./data/inference

  # choose exactly 100 samples
  python sample_xsum.py --input data/xsum_2000.json --num 100 --out_dir ./data

  # reproducible selection
  python sample_xsum.py --input data/xsum_2000.json --percent 5 --seed 42
"""

import argparse
import json
import os
import random
from typing import List

def load_json_or_jsonl(path: str) -> List[dict]:
    """Load either a JSON array file or a JSONL file (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        # Heuristic: if it starts with '[' -> JSON array
        if text[0] == "[":
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON is not a list.")
            return data
        else:
            # try JSONL: parse line by line
            items = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
            return items

def write_json_array(path: str, items: List[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)

def sample_items(items: List[dict], num: int, seed: int = None) -> List[dict]:
    if seed is not None:
        random.seed(seed)
    if num >= len(items):
        # shuffle deterministic if seed given, otherwise keep original order
        if seed is not None:
            items = items[:]  # copy
            random.shuffle(items)
        return items[:num]
    # random.sample returns k unique elements without replacement
    return random.sample(items, num)

def parse_args():
    p = argparse.ArgumentParser(description="Randomly sample subset from a JSON/JSONL dataset.")
    p.add_argument("--input", "-i", required=True, help="Path to input JSON (array) or JSONL file.")
    p.add_argument("--percent", "-p", type=float, default=None,
                   help="Percentage of samples to select (e.g. 5 for 5%%). Mutually exclusive with --num.")
    p.add_argument("--num", "-n", type=int, default=None,
                   help="Exact number of samples to select. Mutually exclusive with --percent.")
    p.add_argument("--out_dir", "-o", default=".", help="Directory to write sampled file (default: current dir).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return p.parse_args()

def main():
    args = parse_args()

    if (args.percent is None) == (args.num is None):
        raise SystemExit("Error: provide exactly one of --percent or --num")

    items = load_json_or_jsonl(args.input)
    total = len(items)
    if total == 0:
        raise SystemExit(f"Input file contains no items: {args.input}")

    if args.percent is not None:
        if not (0 < args.percent <= 100):
            raise SystemExit("Error: --percent must be in (0, 100].")
        num = max(1, int(round(total * args.percent / 100.0)))
    else:
        num = args.num
        if num <= 0:
            raise SystemExit("Error: --num must be > 0")

    if num > total:
        print(f"Requested {num} samples but input has only {total}. Will return all {total} (shuffled if --seed provided).")
        num = total

    sampled = sample_items(items, num, seed=args.seed)

    # create output filename: xsum_train_{num}.json in out_dir
    out_basename = f"xsum_train_{num}.json"
    out_path = os.path.join(args.out_dir, out_basename)
    write_json_array(out_path, sampled)

    print(f"Input: {args.input}")
    print(f"Total items: {total}")
    print(f"Selected: {num} (seed={args.seed})")
    print(f"Wrote sampled file to: {out_path}")

if __name__ == "__main__":
    main()
	
