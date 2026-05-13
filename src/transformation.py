# ============================================================
# FILE: src/transformation.py
# KYA KARTA HAI: PySpark Pipeline se data transform karta hai
#                StringIndexer, OHE, VectorAssembler, Scaler
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import (
    OneHotEncoder, StringIndexer,
    VectorAssembler, StandardScaler
)
from pyspark.ml import Pipeline
import os
import shutil

# 1. Spark Session
spark = SparkSession.builder.appName("Transformation").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Load cleaned data
df = spark.read.csv(
    "data/processed/clean_data",
    header=True,
    inferSchema=True
)
print("Cleaned data loaded!")
print("Rows:", df.count())
print("Columns:", df.columns)

# 3. Categorical columns → StringIndexer + OHE
ohe_cols = ["education_level", "university_tier", "company_type"]

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_Index",
        handleInvalid="keep"
    )
    for c in ohe_cols
]

encoders = [
    OneHotEncoder(
        inputCol=c + "_Index",
        outputCol=c + "_OHE"
    )
    for c in ohe_cols
]

print("StringIndexer + OHE ready.")

# 4. Numerical features assembler + scaler
num_cols = [
    "age", "cgpa", "internships", "projects",
    "programming_languages", "certifications",
    "experience_years", "hackathons", "research_papers",
    "skills_score", "soft_skills_score", "resume_length_words"
]

num_assembler = VectorAssembler(
    inputCols=num_cols,
    outputCol="num_features"
)

scaler = StandardScaler(
    inputCol="num_features",
    outputCol="scaled_num_features",
    withMean=True,
    withStd=True
)

# 5. Final assembler
final_assembler = VectorAssembler(
    inputCols=(
        [c + "_OHE" for c in ohe_cols] +
        ["scaled_num_features"]
    ),
    outputCol="features"
)

# 6. Pipeline
pipeline = Pipeline(
    stages=indexers + encoders + [num_assembler, scaler, final_assembler]
)
pipeline_model = pipeline.fit(df)
df_transformed = pipeline_model.transform(df)
df_final = df_transformed.select("features", "hired")
print("Pipeline done.")

# 7. Train/Test Split
train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)
print("Train rows:", train_df.count())
print("Test  rows:", test_df.count())

# 8. Save as Parquet — PySpark native
output_dir = "data/transformed"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

train_df.write.mode("overwrite").parquet(os.path.join(output_dir, "train"))
test_df.write.mode("overwrite").parquet(os.path.join(output_dir, "test"))
print("Train/Test saved as Parquet.")

# 9. Pipeline save
models_dir = "models/pipeline"
if os.path.exists(models_dir):
    shutil.rmtree(models_dir)
pipeline_model.save(models_dir)
print("Pipeline saved at: models/pipeline")

print("\n===================================")
print("Transformation Complete")
print("data/transformed/train/")
print("data/transformed/test/")
print("models/pipeline/")
print("===================================")

spark.stop()
