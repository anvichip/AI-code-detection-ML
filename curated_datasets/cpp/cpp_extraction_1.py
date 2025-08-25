import pandas as pd
import json

# Data Source: https://github.com/a0ms1n/AI-Code-Detector-for-Competitive-Programming/blob/master/datasets/train.csv
df = pd.read_csv("curated_datasets/cpp/train.csv")

def save_jsonl(sub_df, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for row in sub_df.itertuples():
            f.write(json.dumps({"writer": row.writer, "code": row.source}, ensure_ascii=False) + "\n")
            
sub_df = df.head(1000)
save_jsonl(sub_df, "curated_datasets/cpp/dataset_1.jsonl")