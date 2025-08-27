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
## Results
We chose to merge the three datasets mentioned in the Dataset section together to train our models.

We used `powerset_script.py` to train the models on individual and merged dataset for each language and also a merged dataset.

The results are as follow:
# Sheet1

|Dataset|Model|Train Size|Test Size|Accuracy|Precision|Recall|F1|
|---|---|---|---|---|---|---|---|
|cpp_cpp_dataset_3_cpp_dataset_2|Random Forest|73|19|0.95|0.91|1.0|0.95|
|cpp_cpp_dataset_3_cpp_dataset_2|SVM|73|19|0.53|1.0|0.1|0.18|
|cpp_cpp_dataset_3_cpp_dataset_2|XGBoost|73|19|0.95|0.91|1.0|0.95|
|cpp_cpp_dataset_3_cpp_dataset_2|MLP|73|19|1.0|1.0|1.0|1.0|
|cpp_cpp_dataset_3_cpp_dataset_2|Voting Ensemble|73|19|1.0|1.0|1.0|1.0|
|cpp_cpp_dataset_3|Random Forest|67|17|1.0|1.0|1.0|1.0|
|cpp_cpp_dataset_3|SVM|67|17|0.53|1.0|0.11|0.2|
|cpp_cpp_dataset_3|XGBoost|67|17|1.0|1.0|1.0|1.0|
|cpp_cpp_dataset_3|MLP|67|17|0.94|1.0|0.89|0.94|
|cpp_cpp_dataset_3|Voting Ensemble|67|17|1.0|1.0|1.0|1.0|
|php_php_dataset_2|Random Forest|4|2|0.0|0.0|0.0|0.0|
|php_php_dataset_2|SVM|4|2|0.0|0.0|0.0|0.0|
|php_php_dataset_2|XGBoost|4|2|0.0|0.0|0.0|0.0|
|php_php_dataset_2|MLP|4|2|0.0|0.0|0.0|0.0|
|php_php_dataset_2|Voting Ensemble|4|2|0.0|0.0|0.0|0.0|
|cpp_cpp_dataset_3_cpp_dataset_1_cpp_dataset_2|Random Forest|873|219|0.92|0.94|0.91|0.92|
|cpp_cpp_dataset_3_cpp_dataset_1_cpp_dataset_2|SVM|873|219|0.81|0.77|0.92|0.84|
|cpp_cpp_dataset_3_cpp_dataset_1_cpp_dataset_2|XGBoost|873|219|0.95|0.95|0.97|0.96|
|cpp_cpp_dataset_3_cpp_dataset_1_cpp_dataset_2|MLP|873|219|0.93|0.98|0.89|0.93|
|cpp_cpp_dataset_3_cpp_dataset_1_cpp_dataset_2|Voting Ensemble|873|219|0.95|0.96|0.94|0.95|
|java_java_dataset_3|Random Forest|67|17|1.0|1.0|1.0|1.0|
|java_java_dataset_3|SVM|67|17|0.47|0.0|0.0|0.0|
|java_java_dataset_3|XGBoost|67|17|0.88|1.0|0.78|0.88|
|java_java_dataset_3|MLP|67|17|1.0|1.0|1.0|1.0|
|java_java_dataset_3|Voting Ensemble|67|17|1.0|1.0|1.0|1.0|
|cpp_cpp_dataset_3_cpp_dataset_1|Random Forest|867|217|0.91|0.92|0.9|0.91|
|cpp_cpp_dataset_3_cpp_dataset_1|SVM|867|217|0.81|0.77|0.91|0.83|
|cpp_cpp_dataset_3_cpp_dataset_1|XGBoost|867|217|0.92|0.89|0.96|0.92|
|cpp_cpp_dataset_3_cpp_dataset_1|MLP|867|217|0.88|0.94|0.83|0.88|
|cpp_cpp_dataset_3_cpp_dataset_1|Voting Ensemble|867|217|0.93|0.92|0.95|0.93|
|python_python_dataset_3|Random Forest|67|17|0.94|1.0|0.89|0.94|
|python_python_dataset_3|SVM|67|17|0.47|0.0|0.0|0.0|
|python_python_dataset_3|XGBoost|67|17|0.88|1.0|0.78|0.88|
|python_python_dataset_3|MLP|67|17|0.94|1.0|0.89|0.94|
|python_python_dataset_3|Voting Ensemble|67|17|0.94|1.0|0.89|0.94|
|cpp_cpp_dataset_1_cpp_dataset_2|Random Forest|806|202|0.93|0.93|0.93|0.93|
|cpp_cpp_dataset_1_cpp_dataset_2|SVM|806|202|0.87|0.86|0.9|0.88|
|cpp_cpp_dataset_1_cpp_dataset_2|XGBoost|806|202|0.94|0.94|0.94|0.94|
|cpp_cpp_dataset_1_cpp_dataset_2|MLP|806|202|0.93|0.9|0.97|0.94|
|cpp_cpp_dataset_1_cpp_dataset_2|Voting Ensemble|806|202|0.94|0.94|0.94|0.94|
|cpp_cpp_dataset_1|Random Forest|800|200|0.91|0.93|0.9|0.92|
|cpp_cpp_dataset_1|SVM|800|200|0.86|0.85|0.9|0.87|
|cpp_cpp_dataset_1|XGBoost|800|200|0.92|0.93|0.92|0.93|
|cpp_cpp_dataset_1|MLP|800|200|0.95|0.95|0.95|0.95|
|cpp_cpp_dataset_1|Voting Ensemble|800|200|0.92|0.93|0.93|0.93|
|cpp_cpp_dataset_2|Random Forest|6|2|0.5|1.0|0.5|0.67|
|cpp_cpp_dataset_2|SVM|6|2|0.0|0.0|0.0|0.0|
|cpp_cpp_dataset_2|XGBoost|6|2|0.0|0.0|0.0|0.0|
|cpp_cpp_dataset_2|MLP|6|2|0.5|1.0|0.5|0.67|
|cpp_cpp_dataset_2|Voting Ensemble|6|2|0.5|1.0|0.5|0.67|
|java_java_dataset_3_java_dataset_2|Random Forest|76|20|0.9|1.0|0.82|0.9|
|java_java_dataset_3_java_dataset_2|SVM|76|20|0.65|1.0|0.36|0.53|
|java_java_dataset_3_java_dataset_2|XGBoost|76|20|0.9|1.0|0.82|0.9|
|java_java_dataset_3_java_dataset_2|MLP|76|20|0.95|1.0|0.91|0.95|
|java_java_dataset_3_java_dataset_2|Voting Ensemble|76|20|0.9|1.0|0.82|0.9|
|go_go_dataset_2|Not Processed|1|1|Not Processed|Not Processed|Not Processed|Not Processed|
|python_python_dataset_3_python_dataset_2|Random Forest|83|21|0.95|1.0|0.86|0.92|
|python_python_dataset_3_python_dataset_2|SVM|83|21|0.33|0.33|1.0|0.5|
|python_python_dataset_3_python_dataset_2|XGBoost|83|21|1.0|1.0|1.0|1.0|
|python_python_dataset_3_python_dataset_2|MLP|83|21|1.0|1.0|1.0|1.0|
|python_python_dataset_3_python_dataset_2|Voting Ensemble|83|21|1.0|1.0|1.0|1.0|
|java_java_dataset_2|Random Forest|9|3|0.0|0.0|0.0|0.0|
|java_java_dataset_2|SVM|9|3|0.0|0.0|0.0|0.0|
|java_java_dataset_2|XGBoost|9|3|0.0|0.0|0.0|0.0|
|java_java_dataset_2|MLP|9|3|0.67|1.0|0.67|0.8|
|java_java_dataset_2|Voting Ensemble|9|3|0.33|1.0|0.33|0.5|
|python_python_dataset_2|Random Forest|16|4|0.75|0.5|1.0|0.67|
|python_python_dataset_2|SVM|16|4|0.25|0.25|1.0|0.4|
|python_python_dataset_2|XGBoost|16|4|0.25|0.0|0.0|0.0|
|python_python_dataset_2|MLP|16|4|0.75|0.5|1.0|0.67|
|python_python_dataset_2|Voting Ensemble|16|4|0.5|0.0|0.0|0.0|
|javascript_javascript_dataset_2|Random Forest|38|10|0.8|0.67|1.0|0.8|
|javascript_javascript_dataset_2|SVM|38|10|0.6|0.5|1.0|0.67|
|javascript_javascript_dataset_2|XGBoost|38|10|0.8|0.67|1.0|0.8|
|javascript_javascript_dataset_2|MLP|38|10|0.8|0.67|1.0|0.8|
|javascript_javascript_dataset_2|Voting Ensemble|38|10|0.8|0.67|1.0|0.8|

## Convert to tsv file
Optionally, you can choose to convert your results into a .tsv for easier readability by using `tsv_creator_script.py`

```
python3 -m scripts.tsv_creator_script --data_dir path/to/your/results/directory --metrics_file codebert_metrics.json --output_tsc name_to_save_tsv_file
```
