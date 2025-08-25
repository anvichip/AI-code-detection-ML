import json

# Dataset Source: https://github.com/Back3474/AI-Human-Generated-Program-Code-Dataset/tree/main

input_file = "AI-Human-Generated-Program-Code-Dataset.json" 
data_file = "curated_datasets/python/dataset_3.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data_list = json.load(f) 

with open(data_file, "w", encoding="utf-8") as data_file:
    for item in data_list:
        if "language" in item:
            if item["language"] == 'Python':
                if "ai_generated_code" in item:
                    data_file.write(json.dumps({"code": item["ai_generated_code"], "writer": "AI"}, ensure_ascii=False) + "\n")
                if "human_generated_code" in item:
                    data_file.write(json.dumps({"code": item["human_generated_code"], "writer": "Human"}, ensure_ascii=False) + "\n")
