# ============================================================
# FILE: src/best_model.py
# KYA KARTA HAI: Saare trained models mein se best model
#                dhundta hai aur use "Production" mein
#                register karta hai DagsHub pe
#
# SIMPLE MISAAL:
#   4 contestants ke baad judge final winner choose karta hai
#   aur use trophy deta hai. Yahi kaam yeh file karti hai.
#
# INPUT:  MLflow experiment runs (DagsHub pe)
# OUTPUT: Best model → "Production" status mein register
#         data_and_model/best_model_info.txt
# ============================================================

import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USER     = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri("mlruns")
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

EXPERIMENT_NAME = "resume-hiring-prediction"

def find_best_model():
    """
    MLflow mein se best F1 score wala model dhundho.
    F1 score isliye use karte hain kyunki yeh precision
    aur recall dono ka balance deta hai.
    """
    print("=" * 50)
    print("Step 4: Best Model Dhundha Ja Raha Hai...")
    print("=" * 50)

    client = MlflowClient()

    # Experiment ID dhundo
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise Exception(f"Experiment '{EXPERIMENT_NAME}' nahi mila! Pehle train.py chalao.")

    # Saare runs lao, F1 se sort karo
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"]
    )

    if not runs:
        raise Exception("Koi run nahi mila! Pehle train.py chalao.")

    best_run = runs[0]
    best_run_id   = best_run.info.run_id
    best_f1       = best_run.data.metrics.get("f1", 0)
    best_accuracy = best_run.data.metrics.get("accuracy", 0)
    best_name     = best_run.data.tags.get("mlflow.runName", "Unknown")

    print(f"Best Model: {best_name}")
    print(f"  F1 Score: {best_f1:.4f}")
    print(f"  Accuracy: {best_accuracy:.4f}")
    print(f"  Run ID:   {best_run_id}")

    return best_run_id, best_name, best_f1, best_accuracy

def register_best_model(run_id, model_name, f1, accuracy):
    """
    Best model ko MLflow Model Registry mein register karo
    aur 'Production' mein move karo.
    
    MATLAB: Ab yeh model officially "approved" hai use karne ke liye.
    """
    client = MlflowClient()
    registered_name = "resume_hiring_best_model"

    # Model register karo
    try:
        model_uri = f"models:/resume_{model_name.lower().replace(' ', '_')}/1"
        mv = mlflow.register_model(model_uri=model_uri, name=registered_name)
    except Exception:
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=registered_name)

    print(f"\nModel registered: {registered_name} (version {mv.version})")

    # Production mein promote karo
    client.transition_model_version_stage(
        name=registered_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model '{registered_name}' v{mv.version} → Production!")

    # Info file save karo (prediction.py aur app.py use karein ge)
    with open("data_and_model/best_model_info.txt", "w") as f:
        f.write(f"model_name={registered_name}\n")
        f.write(f"run_id={run_id}\n")
        f.write(f"f1={f1:.4f}\n")
        f.write(f"accuracy={accuracy:.4f}\n")
        f.write(f"original_name={model_name}\n")
        f.write(f"stage=Production\n")

    print("best_model_info.txt save ho gaya!")
    print("Step 4 Complete!\n")

def select_best():
    run_id, name, f1, accuracy = find_best_model()
    register_best_model(run_id, name, f1, accuracy)

if __name__ == "__main__":
    select_best()
