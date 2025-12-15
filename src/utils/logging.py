import mlflow
import os

def setup_mlflow(experiment_name: str, tracking_uri: str = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)


def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: dict, step: int):
    for k, v in metrics.items():
        mlflow.log_metric(k, v, step=step)


def log_model(model, artifact_path="model"):
    mlflow.pytorch.log_model(model, artifact_path)
