import os
import json

def merge_jsonl_in_dir(root_dir):
    ai_data = []
    human_data = []

    def loop_directory(directory):
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith(".jsonl"):
                print(f"📂 Found file: {entry.path}")
                try:
                    with open(entry.path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                if data.get("writer") == "AI":
                                    ai_data.append(data)
                                elif data.get("writer") == "Human":
                                    human_data.append(data)

                            except json.JSONDecodeError as e:
                                print(f"Error decoding line in {entry.path}: {e}")
                except Exception as e:
                    print(f"Error reading {entry.path}: {e}")
            elif entry.is_dir():
                loop_directory(entry.path)

    loop_directory(root_dir)

    data_path = os.path.join(root_dir, "data_merged.jsonl")
    with open(data_path, "w", encoding="utf-8") as f:
        for obj in human_data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        for obj in ai_data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return data_path

merge_jsonl_in_dir("curated_datasets")