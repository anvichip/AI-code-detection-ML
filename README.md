# AI Code Dectection

This project focuses on training machine learning models to distinguish between human-written and AI-generated code using CodeBERT.

The following machine learning models are trained and used to classify between AI and human generated code.
- MLP
- Random Forest
- SVM
- XGBoost
- Ensemble Model (A soft-voting ensemble that combines Random Forest, SVM, XGBoost, and MLP by averaging their predicted probabilities)

## Installation 
```
git clone https://github.com/anvichip/AI-code-detection-ML.git
cd AI-code-detection-ML
pip install -r requirements.txt
```

## Datasets
The project uses 3 open-source datasets for training and evaluating the models.

Dataset 1: https://github.com/a0ms1n/AI-Code-Detector-for-Competitive-Programming/blob/master/datasets/train.csv  
Dataset 2: https://github.com/zzarif/AI-Detector/tree/main/data/dataset-source-codes  
Dataset 3: https://github.com/Back3474/AI-Human-Generated-Program-Code-Dataset/blob/main/AI-Human-Generated-Program-Code-Dataset.jsonl

File names during training are saved using the same convention of datasets naming.

## Training
Use the following command to run `train_script.py` to train model on a datset to 
```
python3 -m scripts.train_script --dataset path/to/your/dataset.jsonl --language programming languages being used in training --save_dir results 
```
