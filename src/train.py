import optuna

from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def calculate_rmse(model, X, y):
    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)

    return mse ** 0.5


def build_model(model_name, params, random_state):

    if model_name == "decision_tree":

        return DecisionTreeRegressor(
            **params,
            random_state=random_state
        )

    if model_name == "xgboost":

        return XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def suggest_parameters(trial, hyperparameters):

    params = {}

    for name, config in hyperparameters.items():

        parameter_type = config["type"]

        if parameter_type == "int":

            params[name] = trial.suggest_int(
                name,
                config["low"],
                config["high"]
            )

        elif parameter_type == "float":

            params[name] = trial.suggest_float(
                name,
                config["low"],
                config["high"],
                log=config.get("log", False)
            )

        elif parameter_type == "categorical":

            params[name] = trial.suggest_categorical(
                name,
                config["choices"]
            )

        else:
            raise ValueError(
                f"Unknown parameter type: {parameter_type}"
            )

    return params


def optimize_model(
    model_name,
    hyperparameters,
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials,
    random_state,
    trial_callback=None
):

    def objective(trial):

        params = suggest_parameters(
            trial,
            hyperparameters
        )

        model = build_model(
            model_name,
            params,
            random_state
        )

        model.fit(
            X_train,
            y_train
        )

        rmse = calculate_rmse(
            model,
            X_val,
            y_val
        )

        # Let tracking.py record the trial
        if trial_callback is not None:

            trial_callback(
                trial=trial,
                model_name=model_name,
                params=params,
                metrics={
                    "rmse": rmse
                }
            )

        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=random_state
        )
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    # Train final model using best parameters
    best_model = build_model(
        model_name,
        study.best_params,
        random_state
    )

    best_model.fit(
        X_train,
        y_train
    )

    return study, best_model