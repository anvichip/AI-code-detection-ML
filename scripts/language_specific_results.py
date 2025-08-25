#!/usr/bin/env python3
import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Run train_script.py on multiple datasets inside curated_datasets folder."
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["cpp", "go", "java", "javascript", "php", "python"],
        help="Programming languages to include (default: all supported)."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="curated_datasets",
        help="Base directory containing language folders."
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default="train_script.py",
        help="Path to the training script."
    )
    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Base directory '{args.base_dir}' not found.")
        return

    for lang in args.languages:
        lang_dir = os.path.join(args.base_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"⚠️ Skipping {lang}: folder not found.")
            continue

        for file_name in sorted(os.listdir(lang_dir)):
            if not file_name.endswith(".jsonl"):
                continue

            dataset_path = os.path.join(lang_dir, file_name)
            print(f"\nTraining on {dataset_path}")

            cmd = ["python3", args.train_script, "--dataset", dataset_path, "--language", lang]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Training failed for {dataset_path}: {e}")

if __name__ == "__main__":
    main()
