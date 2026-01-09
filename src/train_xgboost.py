import json
import numpy as np
import pandas as pd

from data.preprocess import load_and_preprocess_data
from models.xgboost_model import build_xgboost_model
from evaluation.metrics import classification_metrics, stability_physics_metrics


def main():
    # Load and preprocess
    # X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(
    #     "../datasets/power_system_dataset.csv"
    # )
    import os

    DATASET_PATH = os.getenv(
        "DATASET_PATH",
        "../datasets/power_system_dataset.csv"  # local default
    )

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(
        DATASET_PATH
    )

    # Build model
    model = build_xgboost_model()

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Load original data for physics metrics
    # df = pd.read_csv("../datasets/power_system_dataset.csv").iloc[y_test.index]
    df_full = pd.read_csv(DATASET_PATH)
    df = df_full.iloc[y_test.index]

    # Metrics
    clf_metrics = classification_metrics(y_test, y_pred, y_prob)
    phys_metrics = stability_physics_metrics(df, y_pred)

    results = {
        "classification_metrics": clf_metrics,
        "stability_physics_metrics": phys_metrics
    }

    # Save results
    os.makedirs("../results/xgboost", exist_ok=True)

    with open("../results/xgboost/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    np.save("../results/xgboost/confusion_matrix.npy",
            np.array([[sum((y_test == 0) & (y_pred == 0)),
                       sum((y_test == 0) & (y_pred == 1))],
                      [sum((y_test == 1) & (y_pred == 0)),
                       sum((y_test == 1) & (y_pred == 1))]]))

    print("XGBoost training completed successfully.")
    print(results)


if __name__ == "__main__":
    main()
