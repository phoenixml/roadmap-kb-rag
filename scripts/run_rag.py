from __future__ import annotations
import argparse, json
from reason.self_rag import run_self_rag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    args = ap.parse_args()
    res = run_self_rag(args.question)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
