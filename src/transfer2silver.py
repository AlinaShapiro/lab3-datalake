import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from delta import configure_spark_with_delta_pip

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/silver_loader.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("silver_loader")

builder = SparkSession.builder \
    .appName("Transfer2Silver") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

try:
    bronze_path = "file:///app/data/bronze"
    silver_path = "file:///app/data/silver"
    os.makedirs("/app/data/silver", exist_ok=True)

    logger.info(f"Loading the data from Bronze at {bronze_path}")
    df_bronze = spark.read.format("delta").load(bronze_path)

    logger.info("The data is successfully loaded from Bronze.")

    logger.info("Data cleaning...")
    df_silver = df_bronze \
        .filter(col("Quantity").isNotNull() & col("Price").isNotNull()) \
        .withColumn("TotalSpending", col("Quantity") * col("Price")) \
        .groupBy("CustomerID") \
        .agg(
            {"TotalSpending": "sum", "Invoice": "count"}
        ) \
        .withColumnRenamed("sum(TotalSpending)", "TotalSpending") \
        .withColumnRenamed("count(Invoice)", "PurchaseFrequency")

    logger.info("The data is cleaned and aggregated for the Silver Layer")

    df_silver.write.format("delta").mode("overwrite").save(silver_path)
    logger.info(f"Data successfully saved to Silver layer.")
except Exception as e:
    logger.error(f"Error during Silver load:: {e}", exc_info=True)
finally:
    spark.stop()
    logger.info("Spark session stopped.")
