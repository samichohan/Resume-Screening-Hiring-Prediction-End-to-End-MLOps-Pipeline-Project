# ============================================================
# FILE: src/load.py
# KYA KARTA HAI: PySpark se raw data load karke verify karta hai
# ============================================================

from pyspark.sql import SparkSession
import os

# 1. Create Spark Session
spark = SparkSession.builder \
    .appName("Resume Hiring ETL") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("Spark Session Created")

# 2. Path to data
file_path = os.path.join("data/raw", "data.csv")
print("Reading file from:", file_path)

# 3. Load CSV into Spark DataFrame
df = spark.read.csv(file_path, header=True, inferSchema=True)

# 4. Show data
print("\nFirst 5 rows:")
df.show(5)

# 5. Schema check
print("\nSchema:")
df.printSchema()

print("\n===================================")
print("Total Rows   :", df.count())
print("Total Columns:", len(df.columns))
print("===================================")

spark.stop()
