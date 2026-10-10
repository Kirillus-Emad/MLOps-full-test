import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

experiment = client.get_experiment_by_name("Profit Prediction")

print(experiment.experiment_id)
print(experiment.lifecycle_stage)

if experiment.lifecycle_stage == "deleted":
    client.restore_experiment(experiment.experiment_id)
    print("Experiment restored")