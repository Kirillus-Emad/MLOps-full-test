import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

from src.config import load_config

from src.train import optimize_model

from src.evaluate import (
    evaluate_model,
    plot_actual_vs_predicted
)

from src.tracking import (
    setup_mlflow,
    log_optuna_trial,
    start_model_run,
    log_best_model
)

from src.register import register_model


def load_data(config, processed_dir_override=None):

    data_config = config["data"]

    processed_dir = processed_dir_override or data_config["processed_dir"]

    train_path = (
        f"{processed_dir}/"
        f"{data_config['output']['train']}"
    )

    test_path = (
        f"{processed_dir}/"
        f"{data_config['output']['test']}"
    )

    train_df = pd.read_parquet(
        train_path
    )

    test_df = pd.read_parquet(
        test_path
    )

    target = data_config["target"]

    X_train_full = train_df.drop(
        columns=[target]
    )

    y_train_full = train_df[target]

    X_test = test_df.drop(
        columns=[target]
    )

    y_test = test_df[target]

    return (
        X_train_full,
        y_train_full,
        X_test,
        y_test
    )


def main(processed_dir_override=None):

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    config = load_config()

    data_config = config["data"]
    mlflow_config = config["mlflow"]
    training_config = config["training"]

    random_state = training_config["random_state"]

    # --------------------------------------------------
    # MLflow
    # --------------------------------------------------

    setup_mlflow(
        tracking_uri=mlflow_config["tracking_uri"],
        experiment_name=mlflow_config["experiment_name"]
    )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    (
        X_train_full,
        y_train_full,
        X_test,
        y_test
    ) = load_data(config, processed_dir_override=processed_dir_override)

    # --------------------------------------------------
    # Train / validation split
    # --------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=data_config["split"]["test_size"],
        random_state=random_state
    )

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    model_results = []

    # --------------------------------------------------
    # Train each configured model
    # --------------------------------------------------

    for model_name, model_config in config["models"].items():

        if not model_config["enabled"]:
            continue

        print(
            f"\n{'=' * 50}"
        )

        print(
            f"Training {model_name}"
        )

        print(
            f"{'=' * 50}"
        )

        # ----------------------------------------------
        # Bayesian optimization
        # ----------------------------------------------

        study, model = optimize_model(
            model_name=model_name,
            hyperparameters=model_config["hyperparameters"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=training_config["n_trials"],
            random_state=random_state,

            trial_callback=log_optuna_trial
        )

        # ----------------------------------------------
        # Evaluate final model on untouched test set
        # ----------------------------------------------

        metrics, predictions = evaluate_model(
            model,
            X_test,
            y_test
        )

        # ----------------------------------------------
        # Actual vs predicted plot
        # ----------------------------------------------

        figure = plot_actual_vs_predicted(
            y_test,
            predictions,
            title=f"{model_name} - Actual vs Predicted"
        )

        # ----------------------------------------------
        # MLflow best model run
        # ----------------------------------------------

        with start_model_run(
            model_name
        ):

            model_info = log_best_model(
                model=model,
                model_name=model_name,
                params=study.best_params,
                metrics=metrics,
                n_trials=len(study.trials),
                figure=figure
            )

            run_id = mlflow.active_run().info.run_id

        # ----------------------------------------------
        # Store result
        # ----------------------------------------------

        model_results.append(
            {
                "model_name": model_name,
                "model": model,
                "metrics": metrics,
                "run_id": run_id,
                "model_info": model_info
            }
        )

        print(
            f"\n{model_name} results:"
        )

        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.4f}"
            )

    # --------------------------------------------------
    # Find best model
    # --------------------------------------------------

    best_result = min(
        model_results,
        key=lambda result: result["metrics"]["rmse"]
    )

    best_model_name = best_result["model_name"]
    best_run_id = best_result["run_id"]

    print(
        f"\nBest model: {best_model_name}"
    )

    print(
        f"RMSE: "
        f"{best_result['metrics']['rmse']:.4f}"
    )

    # --------------------------------------------------
    # Register best model
    # --------------------------------------------------

    register_model(
        run_id=best_run_id,
        registered_model_name=mlflow_config["registered_model_name"],
        algorithm_name=best_result["model_name"]
    )


if __name__ == "__main__":
    main()