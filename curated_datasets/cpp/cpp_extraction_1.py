import pandas as pd
import json

# Data Source: https://github.com/a0ms1n/AI-Code-Detector-for-Competitive-Programming/blob/master/datasets/train.csv

# Read CSV
df = pd.read_csv("curated_datasets/cpp/train.csv")

# Function to save only the 'code' key
def save_jsonl(sub_df, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for code in sub_df["source"]:
            f.write(json.dumps({"code": code}, ensure_ascii=False) + "\n")

ai_df = df[df["writer"] == "AI"]
human_df = df[df["writer"] == "Human"]

save_jsonl(ai_df, "curated_datasets/cpp/ai_cpp_1.jsonl")
save_jsonl(human_df, "curated_datasets/cpp/human_cpp_1.jsonl")

print("✅ Files saved: ai_cpp_1.jsonl, human_cpp_1.jsonl in cpp directory")

# --- Dataset Summary ---
print("\n📊 Dataset Summary:")
print(f"AI samples: {len(ai_df)}")
print(f"Human samples: {len(human_df)}")
print(f"Total samples: {len(df)}")