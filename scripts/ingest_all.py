from __future__ import annotations
import argparse, json
from ingest.build_kb import ingest_folder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Folder containing PDF files")
    args = ap.parse_args()
    res = ingest_folder(args.pdf_dir)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
