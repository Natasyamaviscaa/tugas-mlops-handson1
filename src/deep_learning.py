from __future__ import annotations

import argparse
import copy
import tempfile
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common import SEED, common_tags, configure_mlflow, load_split, log_run_context, set_seed


class BinaryMLP(nn.Module):
    def __init__(self, input_size: int, mean: np.ndarray, scale: np.ndarray, dropout: float):
        super().__init__()
        self.register_buffer("feature_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.tensor(scale, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        standardized = (inputs - self.feature_mean) / self.feature_scale
        return self.network(standardized).squeeze(1)


def make_loaders(X_train, y_train, X_val, y_val, batch_size: int, seed: int):
    def dataset(X, y) -> TensorDataset:
        return TensorDataset(
            torch.tensor(X.to_numpy(dtype=np.float32)),
            torch.tensor(y.to_numpy(dtype=np.float32)),
        )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(dataset(X_val, y_val), batch_size=batch_size)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn) -> dict[str, float]:
    model.eval()
    losses, labels, probabilities = [], [], []
    with torch.no_grad():
        for features, batch_labels in loader:
            logits = model(features)
            losses.append(loss_fn(logits, batch_labels).item() * len(batch_labels))
            labels.extend(batch_labels.numpy())
            probabilities.extend(torch.sigmoid(logits).numpy())
    predictions = np.asarray(probabilities) >= 0.5
    return {
        "val_loss": sum(losses) / len(loader.dataset),
        "val_f1": f1_score(labels, predictions),
        "val_auc": roc_auc_score(labels, probabilities),
    }


def train_model(
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    dropout: float = 0.2,
    seed: int = SEED,
    metric_logger: Callable[[dict[str, float], int], None] | None = None,
):
    set_seed(seed)
    X_train, X_val, y_train, y_val = load_split(seed)
    scaler = StandardScaler().fit(X_train)
    model = BinaryMLP(X_train.shape[1], scaler.mean_, scaler.scale_, dropout)
    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val, batch_size=batch_size, seed=seed
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = nn.BCEWithLogitsLoss()
    best_f1, best_state, history = -1.0, None, []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        metrics = {
            "train_loss": train_loss,
            **evaluate(model, val_loader, loss_fn),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(metrics)
        if metric_logger:
            metric_logger(metrics, epoch)
        if metrics["val_f1"] > best_f1:
            best_f1 = metrics["val_f1"]
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()

    model.load_state_dict(best_state)
    return model, history, (X_train, X_val, y_train, y_val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logging metrik PyTorch per epoch ke MLflow")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("epochs, batch-size, dan lr harus bernilai positif")

    import mlflow
    import mlflow.pytorch
    from mlflow.models import infer_signature

    configure_mlflow()
    with mlflow.start_run(run_name="pytorch-mlp-per-epoch") as run:
        X_train, _, y_train, _ = load_split()
        log_run_context(X_train, y_train)
        mlflow.set_tags(common_tags(stage="deep-learning", candidate=False))
        mlflow.log_params(
            {
                "architecture": "mlp-30-64-1",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "dropout": args.dropout,
                "seed": SEED,
            }
        )
        model, history, (_, X_val, _, _) = train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            dropout=args.dropout,
            metric_logger=lambda metrics, step: mlflow.log_metrics(metrics, step=step),
        )
        best_epoch = max(range(len(history)), key=lambda index: history[index]["val_f1"])
        best = history[best_epoch]
        mlflow.log_metrics(
            {
                "best_val_f1": best["val_f1"],
                "best_val_auc": best["val_auc"],
                "val_f1": best["val_f1"],
                "val_auc": best["val_auc"],
            }
        )
        mlflow.log_param("best_epoch", best_epoch)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = f"{directory}/best.pt"
            torch.save(model.state_dict(), checkpoint)
            mlflow.log_artifact(checkpoint, artifact_path="checkpoints")

        example = X_val.head(3).to_numpy(dtype=np.float32)
        with torch.no_grad():
            outputs = model(torch.tensor(example)).numpy()
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            signature=infer_signature(example, outputs),
            input_example=example,
        )
        mlflow.set_tag("model_uri", model_info.model_uri)

    print(f"Run PyTorch tersimpan: {run.info.run_id}; best epoch={best_epoch}")


if __name__ == "__main__":
    main()

