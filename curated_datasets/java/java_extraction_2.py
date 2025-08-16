import json

# Dataset Source: https://github.com/Back3474/AI-Human-Generated-Program-Code-Dataset/tree/main

input_file = "AI-Human-Generated-Program-Code-Dataset.json" 

ai_file = "curated_datasets/java/ai_java_2.jsonl"
human_file = "curated_datasets/java/human_java_2.jsonl"

# Load the whole JSON array
with open(input_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)  # returns a Python list

ai_count = 0
human_count = 0

# Write AI and Human files
with open(ai_file, "w", encoding="utf-8") as f_ai, \
     open(human_file, "w", encoding="utf-8") as f_human:

    for item in data_list:
        if "language" in item:
            if item["language"] == 'Java':
                if "ai_generated_code" in item:
                    f_ai.write(json.dumps({"code": item["ai_generated_code"], "language": item["language"]}, ensure_ascii=False) + "\n")
                    ai_count += 1
                if "human_generated_code" in item:
                    f_human.write(json.dumps({"code": item["human_generated_code"], "language": item["language"]}, ensure_ascii=False) + "\n")
                    human_count += 1

print("✅ Files saved: ai_java_2.jsonl, human_java_2.jsonl")
# --- Dataset Summary ---
print("\n📊 Dataset Summary:")
print(f"AI code samples written: {ai_count}")
print(f"Human code samples written: {human_count}")
print(f"Total code samples written: {ai_count + human_count}")
