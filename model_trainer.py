from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

# def train_models(X_train, y_train, X_test, y_test, vectorizer=None):
def train_models(X_train, y_train, X_test, y_test, run_path, vectorizer=None):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    }

    
    for model in models.values():
        print("training ", model)
        model.fit(X_train, y_train)

    #     # Get predicted probabilities
    #     if hasattr(model, "predict_proba"):
    #         y_proba = model.predict_proba(X_test)[:, 1]
    #     else:
    #         # Some models (like SVM without probability=True) may not support predict_proba
    #         y_proba = model.decision_function(X_test)
        
    #     # Compute ROC curve and AUC
    #     fpr, tpr, _ = roc_curve(y_test, y_proba)
    #     roc_auc = auc(fpr, tpr)
        
    #     # Plot
    #     plt.plot(fpr, tpr, lw=2, label=f"{model} (AUC = {roc_auc:.2f})")

    # # Plot settings
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve for Different Models')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    ## Train and Save the trained models to a directory
    model_dir = "trained_models"
    full_model_save_path = os.path.join(run_path, model_dir)
    os.makedirs(full_model_save_path, exist_ok=True)

    print(f"Training and saving models to '{full_model_save_path}'...")

    for name, model in models.items():
        print(f"Training {name}...\n")
        model.fit(X_train, y_train)
        # --- Save individual models ---
        model_filename = os.path.join(full_model_save_path, f"{name.replace(' ', '_').lower()}_model.pkl")
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} model to {model_filename}\n")

    ensemble = VotingClassifier(
        estimators=[
            ('rf', models["Random Forest"]),
            ('svm', models["SVM"]),
            ('xgb', models["XGBoost"]),
            ('mlp', models["MLP"])
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    models["Voting Ensemble"] = ensemble

    ensemble_filename = os.path.join(full_model_save_path, "voting_ensemble_model.pkl")
    with open(ensemble_filename, 'wb') as f:
        pickle.dump(ensemble, f)
    print(f"Saved Voting Ensemble model to {ensemble_filename}")

    metrics_dict = {
        name: evaluate_model(model, X_test, y_test)
        for name, model in models.items()
    }

    return models, vectorizer, metrics_dict
