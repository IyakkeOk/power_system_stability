import os
import json
import torch
import pandas as pd
import numpy as np

from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from data.graph_builder import build_graph_from_row
from models.gnn_model import StabilityGNN
from evaluation.metrics import classification_metrics, stability_physics_metrics


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATASET_PATH = os.getenv(
        "DATASET_PATH",
        "../datasets/power_system_dataset.csv"
    )

    df = pd.read_csv(DATASET_PATH)

    # Build graphs
    graphs = [build_graph_from_row(row) for _, row in df.iterrows()]

    # Train/test split
    split = int(0.8 * len(graphs))
    train_graphs = graphs[:split]
    test_graphs = graphs[split:]

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32)

    model = StabilityGNN(in_channels=7).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    model.train()
    for epoch in range(30):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)

            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)

            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    # Metrics
    clf_metrics = classification_metrics(
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob)
    )

    df_test = df.iloc[split:]
    phys_metrics = stability_physics_metrics(df_test, np.array(y_pred))

    results = {
        "classification_metrics": clf_metrics,
        "stability_physics_metrics": phys_metrics
    }

    os.makedirs("../results/gnn", exist_ok=True)
    with open("../results/gnn/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("GNN training completed successfully.")
    print(results)


if __name__ == "__main__":
    main()
