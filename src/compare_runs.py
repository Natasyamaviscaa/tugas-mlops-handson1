from __future__ import annotations

import json

import mlflow

from src.common import EXPERIMENT_NAME, ROOT, configure_mlflow

SELECTION_PATH = ROOT / "selected_run.json"


def best_candidate():
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.candidate = 'true' and metrics.val_auc > 0",
        order_by=["metrics.val_auc DESC", "metrics.val_f1 DESC"],
    )
    runs = runs[runs["tags.model_uri"].notna()] if "tags.model_uri" in runs else runs.iloc[0:0]
    if runs.empty:
        raise RuntimeError("Tidak ada kandidat bermodel. Jalankan pipeline training terlebih dahulu.")
    return runs.iloc[0], runs


def main() -> None:
    configure_mlflow()
    selected, runs = best_candidate()
    columns = [
        "run_id",
        "tags.mlflow.runName",
        "tags.stage",
        "metrics.val_auc",
        "metrics.val_f1",
    ]
    print(runs[[column for column in columns if column in runs]].head(10).to_string(index=False))

    payload = {
        "run_id": selected["run_id"],
        "run_name": selected["tags.mlflow.runName"],
        "model_uri": selected["tags.model_uri"],
        "val_auc": float(selected["metrics.val_auc"]),
    }
    SELECTION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Kandidat terpilih disimpan ke {SELECTION_PATH}: {payload['run_name']}")


if __name__ == "__main__":
    main()

