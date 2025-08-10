import json

input_file = "AI-Human-Generated-Program-Code-Dataset.json"  # your file with the [ ... ] array

ai_file = "ai.jsonl"
human_file = "human.jsonl"

# Load the whole JSON array
with open(input_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)  # returns a Python list

# Write AI and Human files
with open(ai_file, "w", encoding="utf-8") as f_ai, \
     open(human_file, "w", encoding="utf-8") as f_human:

    for item in data_list:
        if "ai_generated_code" in item:
            f_ai.write(json.dumps({"code": item["ai_generated_code"]}, ensure_ascii=False) + "\n")
        if "human_generated_code" in item:
            f_human.write(json.dumps({"code": item["human_generated_code"]}, ensure_ascii=False) + "\n")

print("✅ Files saved: ai.jsonl, human.jsonl")
