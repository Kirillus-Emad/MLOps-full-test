import mlflow
from mlflow.tracking import MlflowClient


def register_model(
    run_id,
    registered_model_name,
    algorithm_name
):

    client = MlflowClient()

    # ----------------------------------------------
    # Get information about the MLflow run
    # ----------------------------------------------

    run = client.get_run(run_id)

    run_name = run.data.tags.get(
        "mlflow.runName"
    )

    # ----------------------------------------------
    # Register model
    # ----------------------------------------------

    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name
    )

    version = registered_model.version

    # ----------------------------------------------
    # Add metadata to model version
    # ----------------------------------------------

    client.set_model_version_tag(
        name=registered_model_name,
        version=version,
        key="run_id",
        value=run_id
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=version,
        key="run_name",
        value=run_name
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=version,
        key="algorithm",
        value=algorithm_name
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=version,
        key="statuss",
        value="champion"
    )

    # ----------------------------------------------
    # Set champion alias
    # ----------------------------------------------

    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=version
    )

    # ----------------------------------------------
    # Print information
    # ----------------------------------------------

    print(
        f"\nRegistered model:"
        f" {registered_model_name}"
    )

    print(
        f"Version: {version}"
    )

    print(
        f"Algorithm: {algorithm_name}"
    )

    print(
        f"Run name: {run_name}"
    )

    print(
        f"Run ID: {run_id}"
    )

    print(
        "Alias: champion"
    )

    return version