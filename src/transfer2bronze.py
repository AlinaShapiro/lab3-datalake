# import pyspark
# print(pyspark.__version__)
import os
import logging
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/bronze_loader.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bronze_loader")

builder = SparkSession.builder \
    .appName("Transfer2Bronze") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.log.level", "WARN") \
    .config("spark.executor.log.level", "WARN")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

try:
    csv_path = "/app/data/source/online+retail+ii/online_retail_II.csv"
    logger.info(f"Loading CSV from: {csv_path}")
    df = spark.read.option("header", "true").csv(csv_path)
    df.show(5)

    bronze_path = "file:///app/data/bronze"
    os.makedirs("/app/data/bronze", exist_ok=True)
    logger.info(f"Saving to Bronze layer at: {bronze_path}")
    df.write.format("delta").mode("overwrite").save(bronze_path)

    logger.info("Data successfully saved to Bronze layer.")
except Exception as e:
    logger.error(f"Error during Bronze load: {e}", exc_info=True)
finally:
    spark.stop()
    logger.info("Spark session stopped.")
