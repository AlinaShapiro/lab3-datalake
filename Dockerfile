FROM bitnami/spark:latest

USER root

RUN apt-get update && apt-get install -y python3-pip

RUN pip install delta-spark mlflow

WORKDIR /app
