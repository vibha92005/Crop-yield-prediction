import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(csv_path):
    """
    Load and preprocess the crop yield dataset.

    Args:
        csv_path (str): Path to the dataset.

    Returns:
        X_train_scaled, X_test_scaled, X_train, X_test,
        y_train, y_test, X_columns
    """

    df = pd.read_csv(csv_path)

    # Drop ID column
    df = df.drop(columns=["ID"])

    # Encode Crop column
    encoder = LabelEncoder()
    df["Crop"] = encoder.fit_transform(df["Crop"])

    # Features and target
    X = df.drop(columns=["Yield (tons/ha)"])
    y = df["Yield (tons/ha)"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature scaling (used for Linear Regression)
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        X_train,
        X_test,
        y_train,
        y_test,
        X.columns
    )
