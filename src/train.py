# ============================================================
# FILE: src/train.py
# KYA KARTA HAI: PySpark ML models train karta hai
#                MLflow se experiments track karta hai
# ============================================================
import os
from dotenv import load_dotenv
load_dotenv()
from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier, 
    GBTClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.spark

# 1. Spark Session
spark = SparkSession.builder.appName("Resume-Hiring-Training").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Load data (from transformation output)
train_df = spark.read.parquet("data/transformed/train")
test_df  = spark.read.parquet("data/transformed/test")
print("Train:", train_df.count())
print("Test :", test_df.count())

# 3. MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('Resume_Hiring_MultiModel_Experiment')

# 4. Evaluator (AUC)
evaluator = BinaryClassificationEvaluator(
    labelCol="hired",
    metricName="areaUnderROC"
)

# 5. Models dictionary
models = {
    "LogisticRegression": {
        "model": LogisticRegression(labelCol="hired", featuresCol="features"),
        "params": {
            "regParam": [0.01, 0.1],
            "elasticNetParam": [0.0, 0.5]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(labelCol="hired", featuresCol="features"),
        "params": {
            "maxDepth": [3, 5, 7],
            "minInstancesPerNode": [1, 5]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(labelCol="hired", featuresCol="features"),
        "params": {
            "numTrees": [50, 100],
            "maxDepth": [5, 10]
        }    
    },
    "GradientBoosting": {
        "model": GBTClassifier(labelCol="hired", featuresCol="features"),
        "params": {
            "maxDepth": [3, 5],
            "maxIter": [20, 50]
        }
    }
}

# 6. Track best model
best_model_name = None
best_auc        = 0.0
best_model      = None

# 7. Training loop
for name, config in models.items():
    print(f"\nTraining {name}...")
    with mlflow.start_run(run_name=name):
        model = config["model"]

        # Build param grid
        paramGrid = ParamGridBuilder()
        for k, v in config["params"].items():
            paramGrid = paramGrid.addGrid(getattr(model, k), v)
        paramGrid = paramGrid.build()

        # Cross validation
        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=3
        )
        cv_model = cv.fit(train_df)
        predictions = cv_model.transform(test_df)
        auc = evaluator.evaluate(predictions)

        # Accuracy
        acc_eval = MulticlassClassificationEvaluator(
            labelCol="hired", metricName="accuracy")
        accuracy = acc_eval.evaluate(predictions)

        # F1
        f1_eval = MulticlassClassificationEvaluator(
            labelCol="hired", metricName="f1")
        f1 = f1_eval.evaluate(predictions)

        # Precision
        prec_eval = MulticlassClassificationEvaluator(
            labelCol="hired", metricName="weightedPrecision")
        precision = prec_eval.evaluate(predictions)

        # Recall
        rec_eval = MulticlassClassificationEvaluator(
            labelCol="hired", metricName="weightedRecall")
        recall = rec_eval.evaluate(predictions)

        # MLflow log
        mlflow.log_param("model", name)
        mlflow.log_metric("AUC",       auc)
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("f1",        f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_param("model_saved", "local")

        print(f"AUC:       {auc:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")

        # Track best
        if auc > best_auc:
            best_auc        = auc
            best_model      = cv_model.bestModel
            best_model_name = name

# 8. Final summary
print("\n===================================")
print("BEST MODEL:", best_model_name)
print("BEST AUC  :", best_auc)
print("===================================")

# 9. Save best model
mlflow.log_param("model_type", name)
mlflow.spark.log_model(best_model, "best_model")
print("\nTraining Completed Successfully")

spark.stop()
