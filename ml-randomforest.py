import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, size
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

if len(sys.argv) < 4:
    print("Usage: spark-submit ml_rf_pipeline.py <parquet_folder_path> <model_output_path> <metrics_output_path>")
    sys.exit(1)

parquet_folder_path = sys.argv[1]
model_output_path = sys.argv[2]
metrics_output_path = sys.argv[3]

spark = SparkSession.builder.appName("ML-RandomForest-FullPipeline").getOrCreate()

df = spark.read.parquet(parquet_folder_path)

df = df.withColumn("created", to_timestamp(col("created")))
df = df.withColumn("updated", to_timestamp(col("updated")))
df = df.withColumn("duration", (col("updated").cast("long") - col("created").cast("long")) / (60 * 60 * 24))
df = df.dropna(subset=["duration"])

df = df.withColumn("num_roads", size(col("roads")))
df = df.withColumn("num_areas", size(col("areas")))
df = df.withColumn("latitude", col("latitude"))
df = df.withColumn("longitude", col("longitude"))

drop_columns = ["jurisdiction_url", "url", "id", "headline", "description", "+ivr_message", "roads", "areas", "schedule"]
df = df.drop(*drop_columns)

categorical_columns = ["event_type", "severity", "status"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in categorical_columns]

feature_cols = ["num_roads", "num_areas", "latitude", "longitude", "event_type_index", "severity_index", "status_index"]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="duration")

pipeline = Pipeline(stages=indexers + [vector_assembler, scaler, rf])

param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 150]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.maxBins, [16, 32]) \
    .build()

evaluator = RegressionEvaluator(labelCol="duration", predictionCol="prediction", metricName="rmse")

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=3)

train, test = df.randomSplit([0.8, 0.2], seed=42)
cv_model = crossval.fit(train)

predictions = cv_model.bestModel.transform(test)

rmse = evaluator.evaluate(predictions)
mae = RegressionEvaluator(labelCol="duration", predictionCol="prediction", metricName="mae").evaluate(predictions)
mse = RegressionEvaluator(labelCol="duration", predictionCol="prediction", metricName="mse").evaluate(predictions)
r2 = RegressionEvaluator(labelCol="duration", predictionCol="prediction", metricName="r2").evaluate(predictions)


print("Random Forest Pipeline Metrics:")
print(f"   RMSE: {rmse}")
print(f"   MAE: {mae}")
print(f"   MSE: {mse}")
print(f"   R² Score: {r2}")


with open(metrics_output_path, "w") as f:
    f.write("Random Forest Pipeline Metrics:\n")
    f.write(f"   RMSE: {rmse:.4f}\n")
    f.write(f"   MAE: {mae:.4f}\n")
    f.write(f"   MSE: {mse:.4f}\n")
    f.write(f"   R² Score: {r2:.4f}\n")

print(f"Metrics written to {metrics_output_path}")


cv_model.bestModel.write().overwrite().save(model_output_path)
print(f"Full pipeline saved to: {model_output_path}")

spark.stop()


