# import os
# import itertools
# import subprocess
# import argparse
# import tempfile
# import shutil
# from collections import defaultdict

# def find_jsonl_files(root_dir):
#     """Recursively find all .jsonl files excluding ones with 'merged' in the name.
#        Group them by language (top-level directory name)."""
#     files_by_lang = defaultdict(list)
#     for dirpath, _, filenames in os.walk(root_dir):
#         for file in filenames:
#             if file.endswith(".jsonl") and "merged" not in file:
#                 lang = os.path.basename(os.path.dirname(os.path.join(dirpath, file)))
#                 files_by_lang[lang].append(os.path.join(dirpath, file))
#     return files_by_lang

# def powerset(iterable):
#     """Return powerset of an iterable (non-empty subsets)."""
#     s = list(iterable)
#     return (combo for r in range(1, len(s) + 1) for combo in itertools.combinations(s, r))

# def combine_jsonl(files, output_file):
#     """Concatenate multiple .jsonl files into one."""
#     with open(output_file, "w", encoding="utf-8") as outfile:
#         for fname in files:
#             with open(fname, "r", encoding="utf-8") as infile:
#                 shutil.copyfileobj(infile, outfile)

# def main(root_dir, train_script):
#     files_by_lang = find_jsonl_files(root_dir)

#     for lang, files in files_by_lang.items():
#         print(f"🔎 Found {len(files)} JSONL files for language: {lang}")

#         for combo in powerset(files):
#             combo_names = [os.path.basename(f) for f in combo]
#             print(f"Processing {lang} combination: {combo_names}")

#             # Create a temporary merged dataset
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmpfile:
#                 dataset_path = tmpfile.name
#             combine_jsonl(combo, dataset_path)

#             # # Call training script
#             cmd = ["python3", train_script, "--dataset", dataset_path, "--language", lang]
#             try:
#                 subprocess.run(cmd, check=True)
#             except subprocess.CalledProcessError as e:
#                 print(f"❌ Training failed for {combo_names} in {lang}: {e}")
#             finally:
#                 print(f"⚠️ Keeping temp dataset for debugging: {dataset_path}")
#             #     os.remove(dataset_path)  # cleanup temp file

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train on powerset of JSONL datasets")
#     parser.add_argument("--root_dir", type=str, required=True, help="Directory containing language folders with .jsonl files")
#     parser.add_argument("--train_script", type=str, required=True, help="Path to training script")

#     args = parser.parse_args()
#     main(args.root_dir, args.train_script)

import os
import itertools
import subprocess
import argparse
import shutil
from collections import defaultdict

def find_jsonl_files(root_dir):
    """Recursively find all .jsonl files excluding ones with 'merged' in the name.
       Group them by language (top-level directory name)."""
    files_by_lang = defaultdict(list)
    # files_by_lang = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".jsonl") and "merged" not in file:
                lang = os.path.basename(os.path.dirname(os.path.join(dirpath, file)))
                files_by_lang[lang].append(os.path.join(dirpath, file))
                # files_by_lang.append(os.path.join(dirpath, file))
    return files_by_lang

def powerset(iterable):
    """Return powerset of an iterable (non-empty subsets)."""
    s = list(iterable)
    # s = []
    return (combo for r in range(1, len(s) + 1) for combo in itertools.combinations(s, r))

def combine_jsonl(files, output_file):
    """Concatenate multiple .jsonl files into one."""
    with open(output_file, "w", encoding="utf-8") as outfile:
        for fname in files:
            with open(fname, "r", encoding="utf-8") as infile:
                shutil.copyfileobj(infile, outfile)

def main(root_dir, train_script):
    # Directory to store superset datasets
    powerset_dir = os.path.join(os.getcwd(), args.powerset_dir)
    os.makedirs(powerset_dir, exist_ok=True)

    files_by_lang = find_jsonl_files(root_dir)
    # print(files_by_lang)
    # powerset_list = powerset(files_by_lang)
    # print(len(list(powerset_list)))

    for language in files_by_lang:
        powerset_list = powerset(files_by_lang[language])
        for paths in list(powerset_list):
            merged_name = ""
            for path in paths:
                merged_name += path.replace("curated_datasets/", "").replace("/", "_").replace('.jsonl',"_")
            merged_name = merged_name[:-1]

    #     merged_name = "_".join(paths.replace("curated_datasets/", "").replace("/", "__").replace('.jsonl',"_") for path in paths)
    #     print(merged_name)
        # "_".join([os.path.basename(path).replace(".jsonl", "") for path in paths])
        # for path in paths: 
        #     print(path)
        #     merged_name = 
        # print(f"🔎 Found {len(paths)} JSONL files for language: {paths}")

        # powerset_list = powerset(files)
        # print(list(powerset_list))
        # for combo in powerset(files):
        #     combo_names = [os.path.basename(f).replace(".jsonl", "") for f in combo]
        #     merged_name = f"{lang}__{'__'.join(combo_names)}.jsonl"
            dataset_path = os.path.join(powerset_dir, merged_name) + ".jsonl"

        #     print(f"📦 Creating merged dataset: {dataset_path}")

        #     # Combine into superset_datasets dir
            combine_jsonl(paths, dataset_path)

            # Call training script
            cmd = ["python3", train_script, "--dataset", dataset_path, "--language", language, "--save_dir", args.save_dir]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Training failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on powerset of JSONL datasets.")
    parser.add_argument("--root_dir", type=str, required=True, 
                        help="Directory containing language folders with .jsonl files.")
    parser.add_argument("--train_script", type=str, required=True, 
                        help="Path to training script.")
    parser.add_argument("--powerset_dir", default="powerset_datasets", 
                        help="Directory to save powerset of available datasets.")
    parser.add_argument("--save_dir", default="powerset_results", 
                        help="Directory to save the training results.")

    args = parser.parse_args()
    main(args.root_dir, args.train_script)