import argparse
import os
import json
import time
from evaluate_models import run_full_evaluation

def split_jsonl_by_writer(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    human_file = os.path.join(output_dir, "human_data.jsonl")
    ai_file = os.path.join(output_dir, "ai_data.jsonl")

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(human_file, "w", encoding="utf-8") as human_out, \
         open(ai_file, "w", encoding="utf-8") as ai_out:

        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue 

            writer = data.get("writer", "").lower()
            if writer == "human":
                human_out.write(json.dumps(data) + "\n")
            elif writer == "ai":
                ai_out.write(json.dumps(data) + "\n")

    return human_file, ai_file


def main():
    parser = argparse.ArgumentParser(description="Run TF-IDF and CodeBERT based model training.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the JSONL file.")
    parser.add_argument("--language", type=str, required=True,
                        help="Programming language of the dataset.")
    parser.add_argument("--save_dir", type=str, default="powerset_results",
                        help="Directory to save the training results.")
    args = parser.parse_args()
    
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return
    
    save_dir = args.save_dir
    train_num_dir = f"{time.strftime("%Y%m%d_%H%M%S")}_{args.language}_{os.path.basename(dataset_path).replace('.jsonl', '')}"
    run_path = os.path.join(save_dir, train_num_dir)
    os.makedirs(os.path.join(save_dir, train_num_dir), exist_ok=True)

    human_code_path, ai_code_path = split_jsonl_by_writer(dataset_path, run_path)

    # metrics_tfidf = run_full_evaluation(
    #     human_path=human_code_path,
    #     ai_path=ai_code_path,
    #     method="tfidf",
    #     run_path=run_path
    # )
    # print("TF-IDF evaluation complete.")

    metrics_codebert = run_full_evaluation(
        human_path=human_code_path,
        ai_path=ai_code_path,
        method="codebert",
        run_path=run_path
    )
    print("CodeBERT evaluation complete.")

    # -------- Save Results for TFI-DF --------
    # tfidf_metrics_to_save = {}
    # for k, v in metrics_tfidf.items():
    #     if k == "__vectorizer__":  
    #         continue
    #     cleaned_metrics = {metric_key: metric_val for metric_key, metric_val in v.items() if metric_key != 'model'}
    #     tfidf_metrics_to_save[k] = cleaned_metrics

    # tfidf_metrics_filename = os.path.join(run_path, "metrics_tfidf.json")
    # with open(tfidf_metrics_filename, "w") as f:
    #     json.dump(tfidf_metrics_to_save, f, indent=4)
    # print(f"TF-IDF metrics saved to {tfidf_metrics_filename}")

    # -------- Save Results for CodeBERT --------
    code_bert_metrics = {}
    for model_name, metrics in metrics_codebert.items():
        if model_name == "__vectorizer__": 
            continue
        cleaned_metrics = {
            metric_name: metric_value
            for metric_name, metric_value in metrics.items()
            if metric_name != 'model'
        }
        code_bert_metrics[model_name] = cleaned_metrics

    codebert_filename = os.path.join(run_path, "codebert_metrics.json")
    with open(codebert_filename, "w") as f:
        json.dump(code_bert_metrics, f, indent=4)
    print(f"Code BERT metrics saved to {codebert_filename}")


if __name__ == "__main__":
    main()

# Run Command Example
# python3 train_script.py --human_dataset path/to/your/human_code.jsonl --ai_dataset path/to/your/ai_code.jsonl