import pandas as pd

from data_preprocessing import preprocess_data
from visualization import (
    correlation_heatmap,
    rainfall_vs_yield,
    temperature_vs_yield,
    average_yield
)

from train_models import (
    train_linear_regression,
    train_random_forest
)

from evaluation import (
    evaluate_model,
    model_comparison,
    actual_vs_predicted
)

from predict import predict_crop_yield


# ============================================================
# Dataset Path
# ============================================================

# Change this path to your dataset location
DATASET_PATH = "../data/crop_yield.csv"

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully")

# ============================================================
# Visualizations
# ============================================================

correlation_heatmap(df)
rainfall_vs_yield(df)
temperature_vs_yield(df)
average_yield(df)

# ============================================================
# Data Preprocessing
# ============================================================

(
    X_train_scaled,
    X_test_scaled,
    X_train,
    X_test,
    y_train,
    y_test,
    columns
) = preprocess_data(DATASET_PATH)

print("Data Preprocessing Completed")

# ============================================================
# Model Training
# ============================================================

lr_model = train_linear_regression(
    X_train_scaled,
    y_train
)

rf_model = train_random_forest(
    X_train,
    y_train
)

# ============================================================
# Predictions
# ============================================================

lr_pred = lr_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

lr_score = evaluate_model(
    "Linear Regression",
    y_test,
    lr_pred
)

rf_score = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

# ============================================================
# Results
# ============================================================

model_comparison(
    lr_score,
    rf_score
)

actual_vs_predicted(
    y_test,
    rf_pred
)

# ============================================================
# Feature Importance
# ============================================================

importance = pd.Series(
    rf_model.feature_importances_,
    index=columns
)

importance = importance.sort_values()

importance.plot(
    kind="barh",
    figsize=(8,5),
    title="Feature Importance"
)

# ============================================================
# Prediction Example
# ============================================================

predict_crop_yield(
    rf_model,
    columns
)

print("\nProject Completed Successfully")
