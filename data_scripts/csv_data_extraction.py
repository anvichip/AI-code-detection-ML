#!/usr/bin/env python3
import argparse
import pandas as pd
import json
import os

def save_jsonl(sub_df, filename, code_col):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for code in sub_df[code_col]:
            f.write(json.dumps({"code": code}, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Extract AI and Human code from CSV and save as seperate JSONL files.")
    parser.add_argument("--csv", type=str, default="train_data_1.csv",
                        help="Path to source CSV file")
    parser.add_argument("--rows", type=int, default=1000,
                        help="Number of rows to extract for AI and Human each")
    parser.add_argument("--writer-col", type=str, default="writer",
                        help="Column name indicating writer (AI/Human)")
    parser.add_argument("--code-col", type=str, default="source",
                        help="Column name containing the code")
    parser.add_argument("--ai-label", type=str, default="AI",
                        help="Label used in writer-col for AI code")
    parser.add_argument("--human-label", type=str, default="Human",
                        help="Label used in writer-col for Human code")
    parser.add_argument("--ai-out", type=str, default="dataset/ai_1.jsonl",
                        help="Destination JSONL file for AI written code")
    parser.add_argument("--human-out", type=str, default="dataset/human_1.jsonl",
                        help="Destination JSONL file for Human written code")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    ai_df = df[df[args.writer_col] == args.ai_label].head(args.rows)
    human_df = df[df[args.writer_col] == args.human_label].head(args.rows)

    save_jsonl(ai_df, args.ai_out, args.code_col)
    save_jsonl(human_df, args.human_out, args.code_col)

    print("✅ Files saved:")
    print(f"   {args.ai_out}")
    print(f"   {args.human_out}\n")
    print("📊 Dataset Summary:")
    print(f"   AI samples saved: {len(ai_df)}")
    print(f"   Human samples saved: {len(human_df)}")
    print(f"   Total samples in source CSV: {len(df)}")

if __name__ == "__main__":
    main()
