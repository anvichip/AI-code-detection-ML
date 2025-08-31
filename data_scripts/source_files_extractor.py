#!/usr/bin/env python3
import os
import json
import argparse

def generate_dataset(source_dir="dataset-source-codes", output_dir="curated_datasets/cpp", code_extensions=None):
    """
    Generates two JSONL datasets (human and AI code) from a source directory.

    Args:
        source_dir (str): Path to the directory containing source_code_XXX folders.
        output_dir (str): Path to store the generated JSONL datasets.
        code_extensions (list): List of code file extensions to include.
    """
    human_data = []
    ai_data = []

    if code_extensions is None:
        code_extensions = ["cpp", "jav", "php", "py","js","go"] ## Add/Remove extensions as needed

    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each folder in the source directory
    for folder_name in sorted(os.listdir(source_dir)):
        folder_path = os.path.join(source_dir, folder_name)

        if not os.path.isdir(folder_path) or not folder_name.startswith("source_code_"):
            continue

        folder_id_str = folder_name.replace("source_code_", "")

        for file_name in os.listdir(folder_path):
            base_name, ext = os.path.splitext(file_name)
            ext = ext[1:]  # Remove leading dot

            if ext not in code_extensions:
                continue

            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_content = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            entry = {"code": code_content}

            # Human-written code
            if base_name == f"source_code_{folder_id_str}":
                entry["class"] = 0
                entry["model"] = "Human"
                human_data.append(entry)

            # AI Code (GPT-4-Turbo)
            elif f"source_code_{folder_id_str}_gpt-4-turbo_00" in base_name:
                entry["class"] = 1
                entry["model"] = "GPT-4-Turbo"
                ai_data.append(entry)

    # Write to JSONL files
    human_file = os.path.join(output_dir, f"human_{code_extensions[0]}_2.jsonl")
    ai_file = os.path.join(output_dir, f"ai_{code_extensions[0]}_2.jsonl")

    with open(human_file, "w", encoding="utf-8") as f_human:
        for item in human_data:
            f_human.write(json.dumps(item) + "\n")

    with open(ai_file, "w", encoding="utf-8") as f_ai:
        for item in ai_data:
            f_ai.write(json.dumps(item) + "\n")

    print("\n--- Dataset Generation Summary ---")
    print(f"Human code samples written: {len(human_data)}")
    print(f"AI code samples written: {len(ai_data)}")
    print(f"Datasets created: {human_file}, {ai_file}")
    print("----------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Generate human vs AI code datasets in JSONL format.")
    parser.add_argument("--source_dir", type=str, default="dataset-source-codes", help="Path to source dataset directory.")
    parser.add_argument("--output_dir", type=str, default="curated_datasets/cpp", help="Directory to save JSONL datasets.")
    parser.add_argument("--extensions", type=str, nargs="+", default=["cpp"], help="List of code extensions to include.")

    args = parser.parse_args()

    generate_dataset(source_dir=args.source_dir, output_dir=args.output_dir, code_extensions=args.extensions)


if __name__ == "__main__":
    main()
