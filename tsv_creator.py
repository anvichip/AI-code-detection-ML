import os
import json
import csv

def jsonl_line_count(filepath):
    count = 0
    if not os.path.exists(filepath):
        return 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

def process_datasets(root_dir, metrics_file, output_tsv):
    rows = []

    for dataset in os.listdir(root_dir):
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        
        ai_file = os.path.join(dataset_path, "ai_data.jsonl")
        human_file = os.path.join(dataset_path, "human_data.jsonl")

        ai_count = jsonl_line_count(ai_file)
        human_count = jsonl_line_count(human_file)
        dataset_size = ai_count + human_count

        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size

        metrics_path = os.path.join(dataset_path, metrics_file)
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            for model, results in metrics.items():
                print(f"Processing {dataset} with model {model}")
                rows.append({
                    "Dataset": dataset,
                    "Model": model,
                    "Train Size": train_size,
                    "Test Size": test_size,
                    "Accuracy": results.get("accuracy", 0.0),
                    "Precision": results.get("precision", 0.0),
                    "Recall": results.get("recall", 0.0),
                    "F1": results.get("f1", 0.0),
                })

        except Exception as e:
            print(f"Could not process metrics for {dataset}: {e}")
            rows.append({
                "Dataset": dataset,
                "Model": "Not Processed",
                "Train Size": train_size,
                "Test Size": test_size,
                "Accuracy": "Not Processed",
                "Precision": "Not Processed",
                "Recall": "Not Processed",
                "F1": "Not Processed",
            })

    file_exists = os.path.exists(output_tsv)
    with open(output_tsv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Dataset", "Model", "Train Size", "Test Size", "Accuracy", "Precision", "Recall", "F1"],
            delimiter="\t"
        )
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(rows)

    print(f"Results successfully appended to {output_tsv}")

if __name__ == "__main__":
    process_datasets(
        root_dir="results",
        metrics_file="codebert_metrics.json",
        output_tsv="dataset_results.tsv"
    )