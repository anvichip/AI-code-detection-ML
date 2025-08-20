import argparse
import os
import json
import time
from evaluate_models import run_full_evaluation

def main():
    parser = argparse.ArgumentParser(description="Run TF-IDF and CodeBERT based model training.")
    parser.add_argument("--human_dataset", type=str, required=True,
                        help="Path to the Human Code JSONL file.")
    parser.add_argument("--ai_dataset", type=str, required=True,
                        help="Path to the AI Code JSONL file.")
    
    args = parser.parse_args()
    human_code_path = args.human_dataset
    ai_code_path = args.ai_dataset

    if not os.path.exists(human_code_path):
        print(f"Error: Human code file not found at {human_code_path}")
        return
    if not os.path.exists(ai_code_path):
        print(f"Error: AI code file not found at {ai_code_path}")
        return

    print("Starting Training of Models")
    save_dir = "results"
    train_num_dir = time.strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(save_dir, train_num_dir)
    os.makedirs(os.path.join(save_dir, train_num_dir), exist_ok=True)

    metrics_tfidf = run_full_evaluation(
        human_path=human_code_path,
        ai_path=ai_code_path,
        method="tfidf",
        run_path=run_path
    )
    print("TF-IDF evaluation complete.")

    metrics_codebert = run_full_evaluation(
        human_path=human_code_path,
        ai_path=ai_code_path,
        method="codebert",
        run_path=run_path
    )
    print("CodeBERT evaluation complete.")

    # -------- Save Results for TFI-DF --------
    tfidf_metrics_to_save = {}
    for k, v in metrics_tfidf.items():
        if k == "__vectorizer__":  
            continue
        cleaned_metrics = {metric_key: metric_val for metric_key, metric_val in v.items() if metric_key != 'model'}
        tfidf_metrics_to_save[k] = cleaned_metrics

    tfidf_metrics_filename = os.path.join(run_path, "metrics_tfidf.json")
    with open(tfidf_metrics_filename, "w") as f:
        json.dump(tfidf_metrics_to_save, f, indent=4)
    print(f"TF-IDF metrics saved to {tfidf_metrics_filename}")

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