# ============================================================
# FILE: src/cleaning.py
# KYA KARTA HAI: PySpark se raw data saaf karta hai
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim

# 1. Spark Session
spark = SparkSession.builder.appName("Cleaning").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Load dataset
df = spark.read.csv("data/raw/data.csv", header=True, inferSchema=True)
print("===================================")
print("Original Dataset Shape")
print("Rows:", df.count())
print("Columns:", len(df.columns))
print("===================================")

# 3. Show null values count
print("\nChecking Null Values...")
for column in df.columns:
    null_count = df.filter(col(column).isNull()).count()
    if null_count > 0:
        print(f"{column}: {null_count} null values")
print("Null check complete!")

# 4. Remove duplicates
before_duplicates = df.count()
df = df.dropDuplicates()
after_duplicates = df.count()
print("\n===================================")
print("Duplicate Removal")
print("Rows Before:", before_duplicates)
print("Rows After :", after_duplicates)
print("Duplicates Removed:", before_duplicates - after_duplicates)
print("===================================")

# 5. Remove unnecessary column
print("\nRemoving unnecessary column: candidate_id")
df = df.drop("candidate_id")
print("candidate_id column removed successfully.")

# 6. Fill missing numeric values with median
print("\nHandling missing numeric values...")
numeric_cols = ["age", "cgpa", "internships", "projects",
                "programming_languages", "certifications",
                "experience_years", "hackathons", "research_papers",
                "skills_score", "soft_skills_score", "resume_length_words"]

for column in numeric_cols:
    if column in df.columns:
        median_val = df.approxQuantile(column, [0.5], 0.01)[0]
        df = df.fillna({column: median_val})

# 7. Final null handling
before_nulls = df.count()
df = df.dropna()
after_nulls = df.count()
print("\n===================================")
print("Null Value Handling")
print("Rows Before:", before_nulls)
print("Rows After :", after_nulls)
print("Null Rows Removed:", before_nulls - after_nulls)
print("===================================")

# 8. Final schema
print("\n===================================")
print("Final Dataset Schema")
print("===================================")
df.printSchema()

# 9. Preview cleaned data
print("\n===================================")
print("Cleaned Dataset Preview")
print("===================================")
df.show(5)

# 10. Save cleaned dataset using PySpark
output_path = "data/processed/clean_data"
print("\nSaving cleaned dataset using PySpark...")
df.write.mode("overwrite").option("header", True).csv(output_path)
print("\n===================================")
print("Cleaned data saved successfully")
print("Path:", output_path)
print("===================================")

spark.stop()
