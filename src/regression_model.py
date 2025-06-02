import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as Fsum, count as Fcount, desc
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
from delta import configure_spark_with_delta_pip

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/full_pipeline.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("full_pipeline")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.WARNING)

builder = SparkSession.builder \
    .appName("EndToEndPipeline") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

try:
    csv_path = "/app/data/source/online+retail+ii/online_retail_II.csv"
    bronze_path = "file:///app/data/bronze"
    silver_path = "file:///app/data/silver"
    gold_path = "file:///app/data/gold"
    mlflow_uri = "file:///app/mlruns"
    
    try:
        logger.info("\n=== BRONZE LAYER ===")
        logger.info(f"Loading raw data from {csv_path}")
        
        os.makedirs("/app/data/bronze", exist_ok=True)
        logger.info(f"Saving to Bronze layer at: {bronze_path}")
        df_raw = spark.read.option("header", "true").csv(csv_path)
        df_raw.show(5)
        
        os.makedirs("/app/data/bronze", exist_ok=True)
        df_raw.write.format("delta") \
            .option("mergeSchema", "true") \
            .mode("overwrite") \
            .save(bronze_path)
        logger.info(f"Data successfully saved to Bronze layer: {bronze_path}")
    except Exception as e:
        logger.error(f"Error during Bronze load: {e}", exc_info=True)

    try:
        logger.info("\n=== SILVER LAYER ===")
        os.makedirs("/app/data/silver", exist_ok=True)
        logger.info(f"Processing data from {bronze_path}")
        
        df_bronze = spark.read.format("delta").load(bronze_path)
        
        df_silver = df_bronze \
        .filter(col("Quantity").isNotNull() & col("Price").isNotNull()) \
        .withColumn("TotalSpending", col("Quantity") * col("Price")) \
        .groupBy("CustomerID") \
        .agg(
            {"TotalSpending": "sum", "Invoice": "count"}
        ) \
        .withColumnRenamed("sum(TotalSpending)", "TotalSpending") \
        .withColumnRenamed("count(Invoice)", "PurchaseFrequency")

        
        logger.info("Silver data sample:")
        df_silver.show(5)
        
        df_silver.write.format("delta").mode("overwrite").save(silver_path)
        logger.info(f"Data successfully saved to Silver layer: {silver_path}")
    except Exception as e:
         logger.error(f"Error during Silver load:: {e}", exc_info=True)

    try:
    
        logger.info("\n=== GOLD LAYER ===")
        logger.info(f"Preparing ML features from {silver_path}")
        
        df_silver = spark.read.format("delta").load(silver_path)

        df_gold = df_silver \
            .select(
                "CustomerID",
                "TotalSpending",
                "PurchaseFrequency"
            ) \
            .orderBy(desc("TotalSpending"))
        
        logger.info("Gold data sample:")
        df_gold.show(5)

        record_count = df_gold.count()
        
        os.makedirs("/app/data/gold", exist_ok=True)
        df_gold.write.format("delta").mode("overwrite").save(gold_path)
        logger.info(f"Gold layer created with {record_count} records")
    except Exception as e:
        logger.error(f"Error during transfer to Gold layer: {e}", exc_info=True)
        raise

    logger.info("\n=== LINEAR REGRESSION MODEL TRAINING ===")
    logger.info("Preparing data for modeling...")
    
    assembler = VectorAssembler(
        inputCols=["PurchaseFrequency"],
        outputCol="features"
    )
    
    df_ml = assembler.transform(df_gold) \
        .select("features", "TotalSpending")
    
    train, test = df_ml.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Train samples: {train.count()}, Test samples: {test.count()}")
    
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("CustomerSpendingPrediction")
    
    with mlflow.start_run():
        logger.info("Training Linear Regression model...")
        
        lr = LinearRegression(
            featuresCol="features",
            labelCol="TotalSpending",
            maxIter=10,
            regParam=0.3,
            elasticNetParam=0.8
        )
        
        model = lr.fit(train)
        predictions = model.transform(test)

        evaluator = RegressionEvaluator(
            labelCol="TotalSpending",
            predictionCol="prediction"
        )
        
        metrics = {
            "rmse": evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}),
            "r2": evaluator.evaluate(predictions, {evaluator.metricName: "r2"}),
            "mae": evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        }
        
        logger.info(f"Model metrics: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.2f}, MAE={metrics['mae']:.2f}")
        
        mlflow.log_params({
            "features": "PurchaseFrequency",
            "target": "TotalSpending",
            "algorithm": "LinearRegression"
        })
        
        mlflow.log_metrics(metrics)
        

        example = train.limit(5).toPandas()
        example["features"] = example["features"].apply(lambda x: x.toArray().tolist())
        
        mlflow.spark.log_model(
            model,
            "spark-model",
            input_example=example,
            registered_model_name="CustomerSpendingPredictor"
        )

        with open("logs/model_metrics.txt", "w") as f:
            f.write(f"RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"R2: {metrics['r2']:.2f}\n")
            f.write(f"MAE: {metrics['mae']:.2f}\n")
        
        logger.info("Model training and logging completed!")

except Exception as e:
    logger.error(f"Pipeline failed: {e}", exc_info=True)
    raise

finally:
    spark.stop()
    logger.info("Spark session stopped")
    logger.info("\n=== PIPELINE COMPLETED ===")