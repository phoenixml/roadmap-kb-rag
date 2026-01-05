from __future__ import annotations
import argparse, json
from eval.run_eval import load_jsonl, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gold_file", required=True)
    args = ap.parse_args()
    pred = load_jsonl(args.pred_file)
    gold = load_jsonl(args.gold_file)
    report = evaluate(pred, gold)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
