import argparse
import os
import pickle
import sys 
from sctokenizer import CppTokenizer
from tokenizer_utils import get_codebert_embedding

# Define a function to load a single pickled model
def load_model(filepath):
    """Loads a pickled model from the given filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def classify_code_snippet(
    tokenizer_type,
    model_name,
    input_code_content,
    model_load_path 
):
   
    result = None
    try:
        if tokenizer_type == "TF-IDF":
            cpp_vectorizer_path = os.path.join(model_load_path, "vectorizer.pkl")
            if not os.path.exists(cpp_vectorizer_path):
                print(f"Error: TF-IDF vectorizer not found at {cpp_vectorizer_path}. Ensure models were trained with TF-IDF.")
                return None
            cpp_vectorizer = load_model(cpp_vectorizer_path)

            if model_name == "Ensemble":
                model_filepath = os.path.join(model_load_path, "voting_ensemble_model.pkl")
            else:
                model_filepath = os.path.join(model_load_path, f"{model_name.lower()}_model.pkl")

            selected_model = load_model(model_filepath)

            cpp_tokenizer = CppTokenizer()
            cpp_tokens = cpp_tokenizer.tokenize(input_code_content)
            cpp_token_values = ' '.join(token.token_value for token in cpp_tokens)

            features = cpp_vectorizer.transform([cpp_token_values])
            prediction = selected_model.predict(features)[0]

        elif tokenizer_type == "CodeBERT":
            # Load CodeBERT specific models
            if model_name == "Ensemble":
                model_filepath = os.path.join(model_load_path, "voting_ensemble_model.pkl") # Assuming CodeBERT ensemble is also named this way, adjust if different
            else:
                model_filepath = os.path.join(model_load_path, f"codebert_{model_name.lower()}_model.pkl")

            selected_model = load_model(model_filepath)

            embedding = get_codebert_embedding(input_code_content).reshape(1, -1)
            prediction = selected_model.predict(embedding)[0]

        else:
            print("Invalid tokenizer type. Choose 'TF-IDF' or 'CodeBERT'.")
            return None

        result = "\U0001F9E0 AI-Generated" if prediction == 1 else "\U0001F464 Human-Written"
        print(f"Prediction Result using {model_name} with {tokenizer_type} tokenizer:")
        print(f"Classification: {result}")
        return result

    except FileNotFoundError as e:
        print(f"Error: Required model or vectorizer file not found. {e}")
        print("Please ensure models have been trained and saved in the specified 'trained_models' directory.")
        return None
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def main_cli():
    parser = argparse.ArgumentParser(description="Classify a code snippet as human-written or AI-generated.")
    parser.add_argument("--tokenizer", type=str, choices=["TF-IDF", "CodeBERT"], required=True,
                        help="Select tokenizer type: 'TF-IDF' (CppTokenizer) or 'CodeBERT'.")
    parser.add_argument("--model", type=str, required=True,
                        help="Select model name (e.g., 'Random_Forest', 'SVM', 'XGBoost', 'MLP', 'Ensemble'). Use underscores for spaces.")
    parser.add_argument("--code_file", type=str,
                        help="Path to a file containing the code snippet. If not provided, read from stdin.")
    parser.add_argument("--trained_models_path", type=str, required=True,
                        help="Path to the directory containing the trained models and vectorizer (e.g., 'results/20250727_154317/trained_models').")

    args = parser.parse_args()

    input_code_content = ""
    if args.code_file:
        if not os.path.exists(args.code_file):
            print(f"Error: Code file not found at {args.code_file}")
            return
        with open(args.code_file, 'r') as f:
            input_code_content = f.read()
    else:
        print("Paste your code snippet below. Press Ctrl+D (Unix/Linux/macOS) or Ctrl+Z then Enter (Windows) when done:")
        input_code_content = sys.stdin.read()
        if not input_code_content.strip(): # Check if input is empty
             print("No code snippet provided. Exiting.")
             return

    # Call the classification function
    classify_code_snippet(
        tokenizer_type=args.tokenizer,
        model_name=args.model,
        input_code_content=input_code_content,
        model_load_path=args.trained_models_path
    )

if __name__ == "__main__":
    main_cli()


## Example usage:
#python3 classify_script.py --tokenizer TF-IDF --model "Random_Forest" --code_file my_code.cpp --trained_models_path results/20250728_103000/trained_models
#python3 classify_script.py --tokenizer CodeBERT --model "Ensemble" --trained_models_path results/20250728_103000/trained_models