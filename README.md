# Retail Customer Spending Prediction Pipeline
This project implements a complete data pipeline for analyzing customer spending patterns from retail transaction data, featuring Delta Lake architecture with Bronze, Silver, and Gold layers, Spark-based ETL for data processing and MLflow pipeline predicting customer spending

## How to Run
1. Build the Spark container with all dependencies, volums and set up the environment.
```bash
docker-compose up --build
```

2. [Optional] Running Individual Components.
For debugging or manual execution, you can run each pipeline stage separately:

```bash
# access the container shell
docker exec -it spark_app bash

# execute pipeline stages manually
python3 /app/src/transfer2bronze.py
python3 /app/src/transfer2silver.py
python3 /app/src/transfer2gold.py
```

3. The end-to-end pipeline is implemented in `regression_model.py`.

**Pipeline Architecture**: CSV Data → Bronze (Raw) → Silver (Cleaned) → Gold (Aggregated) → ML Model
Data Layers

```bash
docker exec -it spark_app bash
python3 /app/src/regression_model.py 
```
4. See pipeline logs(transfer process to layers, training and metrics logs) in `logs` directory.
