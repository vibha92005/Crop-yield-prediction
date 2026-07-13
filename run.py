import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing import preprocess_data
from src.visualization import (
    correlation_heatmap,
    rainfall_vs_yield,
    temperature_vs_yield,
    average_yield
)
from src.train_models import (
    train_linear_regression,
    train_random_forest
)
from src.evaluation import (
    evaluate_model,
    model_comparison,
    actual_vs_predicted
)
from src.predict import predict_crop_yield

# ============================================================
# Dataset Path
# ============================================================

# Change this path according to your dataset location
DATASET_PATH = "data/crop_yield.csv"

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully")

# ============================================================
# Data Visualization
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
    feature_names
) = preprocess_data(DATASET_PATH)

print("Data Preprocessing Completed")

# ============================================================
# Train Models
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
# Evaluate Models
# ============================================================

lr_r2 = evaluate_model(
    "Linear Regression",
    y_test,
    lr_pred
)

rf_r2 = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

# ============================================================
# Compare Models
# ============================================================

model_comparison(
    lr_r2,
    rf_r2
)

# ============================================================
# Actual vs Predicted
# ============================================================

actual_vs_predicted(
    y_test,
    rf_pred
)

# ============================================================
# Feature Importance
# ============================================================

importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
)

importance = importance.sort_values()

plt.figure(figsize=(8,5))
importance.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ============================================================
# Prediction Example
# ============================================================

predict_crop_yield(
    rf_model,
    feature_names
)

print("\nProject Completed Successfully")
