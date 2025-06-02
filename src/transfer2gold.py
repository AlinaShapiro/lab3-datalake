import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, desc
from delta import configure_spark_with_delta_pip

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/gold_loader.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gold_loader")

builder = SparkSession.builder \
    .appName("Transfer2Gold") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

try:
    silver_path = "file:///app/data/silver"
    gold_path = "file:///app/data/gold"
    os.makedirs("/app/data/gold", exist_ok=True)

    logger.info(f"Loading data from Silver layer at {silver_path}")
    df_silver = spark.read.format("delta").load(silver_path)

    logger.info(f"Available columns in Silver: {df_silver.columns}")
    df_silver.show(5)

    logger.info("Creating Gold layer from aggregated metrics")
    df_gold = df_silver \
        .select(
            "CustomerID",
            "TotalSpending",
            "PurchaseFrequency"
        ) \
        .orderBy(desc("TotalSpending"))

    logger.info("Gold layer sample:")
    df_gold.show(5)

    logger.info(f"Writing Gold layer data to {gold_path}")
    df_gold.write.format("delta").mode("overwrite").save(gold_path)

    record_count = df_gold.count()
    logger.info(f"Gold layer created with {record_count} records")

except Exception as e:
    logger.error(f"Error during transfer to Gold layer: {e}", exc_info=True)
    raise
finally:
    spark.stop()
    logger.info("Spark session stopped.")