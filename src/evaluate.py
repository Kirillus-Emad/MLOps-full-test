import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

    return metrics, predictions


def plot_actual_vs_predicted(
    y_test,
    predictions,
    title="Actual vs Predicted"
):

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    # Perfect prediction line
    min_value = min(
        y_test.min(),
        predictions.min()
    )

    max_value = max(
        y_test.max(),
        predictions.max()
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    ax.grid(True)

    fig.tight_layout()

    return fig