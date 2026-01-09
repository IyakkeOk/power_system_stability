import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(csv_path, test_size=0.2, random_state=42):
    """
    Load and preprocess power system stability dataset.
    """

    df = pd.read_csv(csv_path)

    # Encode categorical bus type
    le = LabelEncoder()
    df["BusType"] = le.fit_transform(df["BusType"])

    # Features and target
    X = df.drop(columns=["StabilityLabel"])
    y = df["StabilityLabel"]

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
