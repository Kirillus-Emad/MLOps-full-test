import mlflow
import mlflow.sklearn
import mlflow.xgboost


def setup_mlflow(
    tracking_uri,
    experiment_name
):

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        experiment_name
    )


def log_optuna_trial(
    trial,
    model_name,
    params,
    metrics
):

    with mlflow.start_run(
        nested=True,
        run_name=f"{model_name}_trial_{trial.number}"
    ):

        mlflow.set_tag(
            "run_type",
            "optuna_trial"
        )

        mlflow.set_tag(
            "model",
            model_name
        )

        mlflow.log_params(
            params
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_param(
            "trial_number",
            trial.number
        )


def start_model_run(
    model_name
):

    return mlflow.start_run(
        run_name=f"{model_name}_Best"
    )


def log_best_model(
    model,
    model_name,
    params,
    metrics,
    n_trials,
    figure
):

    mlflow.set_tag(
        "run_type",
        "best_model"
    )

    mlflow.set_tag(
        "model",
        model_name
    )

    mlflow.set_tag(
        "optimization",
        "Bayesian Optimization - Optuna TPE"
    )

    mlflow.log_params(
        params
    )

    mlflow.log_metrics(
        metrics
    )

    mlflow.log_param(
        "n_trials",
        n_trials
    )

    # Log plot
    mlflow.log_figure(
        figure,
        "actual_vs_predicted.png"
    )

    # Log model
    if model_name == "xgboost":

        model_info = mlflow.xgboost.log_model(
            model,
            name="model"
        )

    else:

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
        )

    return model_info