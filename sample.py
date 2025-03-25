from pyspark.sql import SparkSession
import sys

def sample_parquet(df, output_path: str, fraction: float):
    sampled_df = df.sample(fraction=fraction, seed=42)
    sampled_df.write.mode("overwrite").parquet(f'{output_path}/sample-{fraction}')

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ParquetSampler").getOrCreate()
    input = sys.argv[1]
    output = sys.argv[2]
    fractions = [0.1, 0.25, 0.5, 0.75]
    df = spark.read.parquet(input)
    df.cache()
    for fraction in fractions:
        sample_parquet(df, output, fraction)
