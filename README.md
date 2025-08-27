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

## Dataset Format
The project expects the dataset to be used for training in JSONL format.
Here is and example:

```json
{"writer": "AI", "code": "#include <bits/stdc++.h>\nusing namespace std;\n\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    \n    int t;\n    cin >> t;\n    while (t--) {\n        string s;\n        cin >> s;\n        char min_char = *min_element(s.begin(), s.end());\n        size_t pos = s.find(min_char);\n        string a(1, min_char);\n        string b = s.substr(0, pos) + s.substr(pos + 1);\n        cout << a << \" \" << b << \"\\n\";\n    }\n    \n    return 0;\n}\n"}
{"writer": "Human", "code": "#include <bits/stdc++.h>\nusing namespace std;\n\n#define forn(i,n) for(int i=0;i<int(n);i++)\n#define forsn(i,s,n) for(int i=(int)(s);i<(int)(n);i++)\n#define dforsn(i,s,n) for(int i=(int)(n-1);i>=int(s);i--)\n#define si(a) int((a).size())\n#define pb push_back\n#define mp make_pair\n#define all(a) a.begin(),a.end()\n#define fastio ios_base::sync_with_stdio(false); cin.tie(0)\n#define endl '\\n'\n#ifdef LOCAL\n    #define DBG(a) cerr << #a << \"=\" << a << endl\n    #define RAYA cerr << \"----------\" << endl\n#else\n    #define DBG(a)\n    #define RAYA\n#endif\ntypedef vector<int> vi;\ntypedef pair<int,int> pii;\ntypedef long long int tint;\n\nconst int MAXN = 2e3+10;\nint n, a[MAXN];\n\nint mx() {\n int best = 1, cnt = 1;\n forsn(i, 1, n) {\n  if (a[i] >= a[i-1])\n   cnt++;\n  else\n   cnt = 1;\n  best = max(best, cnt);\n }\n return best;\n}\n\nint main() {\n    fastio;\n\n cin >> n;\n forn(i, n) cin >> a[i];\n\n int best = 0;\n forn(i, n) {\n  int start = i;\n  while (i+1 < n && a[i+1] <= a[i]) i++; \n\n  reverse(a+start, a+i+1);\n  best = max(best, mx());\n  reverse(a+start, a+i+1);\n }\n\n cout << best << endl;\n\n    return 0;\n}"}

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
