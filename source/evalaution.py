import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("-"*40)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    return r2


def model_comparison(lr_score, rf_score):

    models = ["Linear Regression", "Random Forest"]

    scores = [lr_score, rf_score]

    plt.figure(figsize=(7,5))

    bars = plt.bar(models, scores)

    plt.ylabel("R² Score")
    plt.title("Model Comparison")

    for bar, score in zip(bars, scores):

        plt.text(
            bar.get_x()+bar.get_width()/2,
            score+0.01,
            f"{score:.3f}",
            ha="center"
        )

    plt.show()


def actual_vs_predicted(y_true, prediction):

    plt.figure(figsize=(7,7))

    plt.scatter(y_true, prediction)

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--"
    )

    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")

    plt.title("Actual vs Predicted Yield")

    plt.show()
