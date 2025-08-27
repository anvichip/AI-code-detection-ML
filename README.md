# AI Code Dectection

This project focuses on training machine learning models to distinguish between human-written and AI-generated code using CodeBERT.

## Installation 
```
git clone https://github.com/anvichip/AI-code-detection-ML.git
cd AI-code-detection-ML
pip install -r requirements.txt
```

## Datasets
The project uses 3 open-source datasets for training and evaluating the models.

Dataset 1: https://github.com/a0ms1n/AI-Code-Detector-for-Competitive-Programming/blob/master/datasets/train.csv  
Dataset 2: https://github.com/zzarif/AI-Detector/tree/main/data/dataset-source-codes.  
Dataset 3: https://github.com/Back3474/AI-Human-Generated-Program-Code-Dataset/blob/main/AI-Human-Generated-Program-Code-Dataset.jsonl

File names during training are saved using the same convention of datasets naming.

## Training
```
python scripts/train_script.py --data path/to/your/data.jsonl --language programming language to be trained
```
