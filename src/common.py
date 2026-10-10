from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "handson-1-breast-cancer"
MODEL_NAME = "handson-1-breast-cancer-clf"
SEED = 42


def configure_mlflow() -> None:
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def source_dataframe() -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame.copy()


def dataset_md5(frame: pd.DataFrame | None = None) -> str:
    frame = source_dataframe() if frame is None else frame
    values = pd.util.hash_pandas_object(frame, index=True).values
    return hashlib.md5(values.tobytes()).hexdigest()


def load_split(seed: int = SEED):
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def common_tags(stage: str, candidate: bool = False) -> dict[str, str]:
    return {
        "stage": stage,
        "candidate": str(candidate).lower(),
        "dataset": "sklearn-breast-cancer",
        "data_md5": dataset_md5(),
        "split_seed": str(SEED),
    }


def log_run_context(X_train, y_train) -> None:
    import mlflow

    mlflow.set_tags(
        {
            "dataset": "sklearn-breast-cancer",
            "data_md5": dataset_md5(),
            "split_seed": str(SEED),
        }
    )
    mlflow.log_input(
        mlflow.data.from_pandas(
            X_train.assign(target=y_train.to_numpy()),
            name="breast-cancer-train",
            targets="target",
        ),
        context="training",
    )
    requirements = ROOT / "requirements.txt"
    if requirements.exists():
        mlflow.log_artifact(requirements)

