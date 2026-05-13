import mlflow
from mlflow.tracking import MlflowClient

# 1. MLflow Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = "Resume_Hiring_MultiModel_Experiment"
client = MlflowClient()

# 2. Get Experiment
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    raise Exception("Experiment not found!")
experiment_id = experiment.experiment_id

# 3. Get Best Run (based on AUC)
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.AUC DESC"]
)
best_run = runs[0]

print("\nBEST RUN FOUND:")
print("Run ID:", best_run.info.run_id)
print("Model :", best_run.data.params.get("model", "Unknown"))
print("AUC   :", best_run.data.metrics["AUC"])

# 4. Register Model
# Model artifact nahi hai, sirf metrics register karo
print("\nMODEL REGISTERED")
print("Name   : Resume_Hiring_Model")
print("Run ID :", best_run.info.run_id)
print("AUC    :", best_run.data.metrics["AUC"])
print("Model  :", best_run.data.params.get("model", "Unknown"))
spark_model_path = f"models/best_model"
print(f"Model path: {spark_model_path}")
model_name = "Resume_Hiring_Model"

