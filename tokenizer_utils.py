from sctokenizer import CppTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
import torch
import numpy as np

# Load CodeBERT once
codebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
codebert_model.to(device)
codebert_model.eval()

# ----------- TF-IDF Tokenization -----------
def tokenize_codes(codes):
    cpp_tokenizer = CppTokenizer()
    tokenized_codes = []

    for code in codes:
        tokens = cpp_tokenizer.tokenize(code)
        token_values = [token.token_value for token in tokens]
        tokenized_codes.append(' '.join(token_values))

    return tokenized_codes

def extract_features(tokenized_codes, max_features=1000, progress_callback=None):
    if progress_callback:
        for i in range(len(tokenized_codes)):
            progress_callback(i + 1, len(tokenized_codes))  # update progress

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(tokenized_codes)
    return X, vectorizer

# ----------- CodeBERT Embeddings -----------
def get_codebert_embedding(code_string):
    inputs = codebert_tokenizer(code_string, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = codebert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def extract_codebert_features(codes, progress_callback=None):
    embeddings = []
    total = len(codes)
    for i, code in enumerate(codes):
        vec = get_codebert_embedding(code)
        embeddings.append(vec)
        if progress_callback:
            progress_callback(i + 1, total)
    return np.array(embeddings)

