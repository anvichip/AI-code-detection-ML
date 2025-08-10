import pandas as pd
import json

# Read CSV
df = pd.read_csv("train.csv")

# Function to save only the 'code' key
def save_jsonl(sub_df, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for code in sub_df["source"]:
            f.write(json.dumps({"code": code}, ensure_ascii=False) + "\n")

# Filter by writer and save
save_jsonl(df[df["writer"] == "AI"].head(300), "ai.jsonl")
save_jsonl(df[df["writer"] == "Human"].head(300), "human.jsonl")

print("✅ Files saved: ai.jsonl, human.jsonl")
