# AI Code Dectection

This project focuses on training machine learning models to distinguish between human-written and AI-generated code using CodeBERT.

The following machine learning models are trained and used to classify between AI and human generated code.
- MLP
- Random Forest
- SVM
- XGBoost
- Ensemble Model
  - A soft-voting ensemble that combines Random Forest, SVM, XGBoost, and MLP by averaging their predicted probabilities

## Installation 
```
git clone https://github.com/anvichip/AI-code-detection-ML.git
cd AI-code-detection-ML
pip install -r requirements.txt
```

## Datasets
The project uses 3 open-source datasets for training and evaluating the models.

- [Dataset 1](https://github.com/a0ms1n/AI-Code-Detector-for-Competitive-Programming)
- [Dataset 2](https://github.com/zzarif/AI-Detector/tree/main)  
- [Dataset 3](https://github.com/Back3474/AI-Human-Generated-Program-Code-Dataset/blob/main)

File names during training are saved using the same convention of datasets naming.

## Dataset Format
The project expects the dataset to be used for training in JSONL format with keys `writer` and `code` indicating the writer and the code respectively.

```json
{"writer": "AI", "code": "#include <bits/stdc++.h>\nusing namespace std;\n\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    \n    int t;\n    cin >> t;\n    while (t--) {\n        string s;\n        cin >> s;\n        char min_char = *min_element(s.begin(), s.end());\n        size_t pos = s.find(min_char);\n        string a(1, min_char);\n        string b = s.substr(0, pos) + s.substr(pos + 1);\n        cout << a << \" \" << b << \"\\n\";\n    }\n    \n    return 0;\n}\n"}
{"writer": "Human", "code": "#include <iostream>\n#include <cmath>\n#include <cstdio>\nusing namespace std;\nint getDown(int n)\n{\n int ans=0;\n while(n>1)\n {\n  ans++;\n  n>>=1;\n }\n return ans;\n}\n#define N 300\nint l[N]={1};\nint main()\n{\n int i,j=2,tmp,n,a,b,p,q;\n tmp=2;\n for(i=2;i<N;i+=2)\n {\n  l[i]=j;\n  if(i==tmp)\n  {\n   tmp<<=1;\n   j++;\n  }\n }\n scanf(\"%d%d%d\",&n,&a,&b);\n p=min(a,b);\n q=max(a,b);\n if(p==q)\n {\n  printf(\"0\\n\");\n }\n else\n {\n  p=p&1?p:p-1;\n  q=q&1?q:q-1;\n  int cha=q-p;\n  //cout<<cha<<\" \"<<getDown(n);\n  if(l[cha]==getDown(n))\n  {\n   printf(\"Final!\\n\");\n  }\n  else\n  {\n   printf(\"%d\\n\",l[cha]);\n  }\n }\n return 0;\n} "}
```
## Training
Use the following command to run `train_script.py` to train model on a JSONL dataset.
```
python3 -m scripts.train_script --dataset path/to/your/dataset.jsonl --language programming languages being used in training --save_dir results 
```

## Classification
Use the following command to run `classify_script.py` to classify between an AI and a human written code using the trained models.
```
python3 -m scripts.classify_script --tokenizer CodeBERT or TF-IDF --model "" --code_file path/to/your/code/file --trained_models_path path/to/your/trained/models
```
## Experiment
### Dataset

### Setup
We chose to merge the three datasets mentioned in the Dataset section together to train our models.
We used `powerset_script.py` to train the models on individual and merged dataset for each language and also a merged dataset.
The complete results can be found here.

### Research Question
- **Question 1**:
- **Question 2**:
- **Question 3**:
- **Question 4**:
  
### Results

## Convert to tsv file
Optionally, you can choose to convert your results into a .tsv for easier readability by using `tsv_creator_script.py`
```
python3 -m scripts.tsv_creator_script --data_dir path/to/your/results/directory --metrics_file codebert_metrics.json --output_tsc name_to_save_tsv_file
```
