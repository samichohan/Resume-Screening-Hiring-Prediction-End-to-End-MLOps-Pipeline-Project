# ============================================================
# FILE: src/train.py
# KYA KARTA HAI: 4 alag alag ML models train karta hai aur
#                har ek ka result MLflow mein save karta hai
#
# SIMPLE MISAAL:
#   4 alag ustads se ek hi sawaal ka jawab maango.
#   Har ustad ki accuracy, speed sab note karo (MLflow).
#   Phir decide karo kaun best hai.
#
# INPUT:  data_and_model/X_train.csv
#         data_and_model/X_test.csv
#         data_and_model/y_train.csv
#         data_and_model/y_test.csv
# OUTPUT: MLflow experiment runs (DagsHub pe visible honge)
#         data_and_model/model_results.csv (comparison table)
# ============================================================

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from dotenv import load_dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ---- ENVIRONMENT VARIABLES LOAD KARO (.env file se) ----
load_dotenv()

# ---- MLFLOW DAGSHUB SE CONNECT KARO ----
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USER     = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("resume-hiring-prediction")

# ---- PARAMS ----
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

CV           = params["model"]["cv"]
RANDOM_STATE = params["data"]["random_state"]

def load_data():
    """Preprocessed data load karo"""
    X_train = pd.read_csv("data_and_model/X_train.csv")
    X_test  = pd.read_csv("data_and_model/X_test.csv")
    y_train = pd.read_csv("data_and_model/y_train.csv").squeeze()
    y_test  = pd.read_csv("data_and_model/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def get_metrics(y_true, y_pred, y_proba):
    """
    Model ki performance measure karo.
    
    METRICS KA MATLAB:
    - Accuracy:  100 mein se kitne sahi predict hue  (e.g. 94%)
    - Precision: Jo "hired" bola, unme se kitne sach mein hired the
    - Recall:    Jo sach mein hired the, unme se kitne pakde
    - F1:        Precision aur Recall ka balance
    - ROC-AUC:   Overall model ki quality (1.0 = perfect)
    """
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba)
    }

def train_and_log(name, model, X_train, X_test, y_train, y_test):
    """
    Ek model train karo aur MLflow mein log karo.
    
    MLflow KYA SAVE KARTA HAI:
    - Model ka naam
    - Settings (hyperparameters)
    - Accuracy, F1 waghaira (metrics)
    - Actual trained model file
    """
    print(f"\nTraining: {name}...")

    with mlflow.start_run(run_name=name):
        # Model train karo
        model.fit(X_train, y_train)

        # Predictions lao
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics calculate karo
        metrics = get_metrics(y_test, y_pred, y_proba)

        # MLflow mein log karo
        mlflow.log_params(model.get_params())   # settings save
        mlflow.log_metrics(metrics)             # scores save
        mlflow.sklearn.log_model(               # model file save
            model,
            artifact_path="model",
            registered_model_name=f"resume_{name.lower().replace(' ', '_')}"
        )

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  MLflow Run ID: {mlflow.active_run().info.run_id}")

    return {**{"model_name": name}, **metrics}

def train_all():
    print("=" * 50)
    print("Step 3: Model Training Shuru...")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # ---- 4 MODELS DEFINE KARO ----
    # Har model ek alag tareeqa hai data se seekhne ka
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=200, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }

    # ---- HAR MODEL KO TRAIN KARO ----
    all_results = []
    for name, model in models.items():
        result = train_and_log(name, model, X_train, X_test, y_train, y_test)
        all_results.append(result)

    # ---- COMPARISON TABLE SAVE KARO ----
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("f1", ascending=False)
    results_df.to_csv("data_and_model/model_results.csv", index=False)

    print("\n" + "=" * 50)
    print("ALL MODELS RESULTS:")
    print("=" * 50)
    print(results_df.to_string(index=False))
    print("\nStep 3 Complete!\n")

if __name__ == "__main__":
    train_all()
