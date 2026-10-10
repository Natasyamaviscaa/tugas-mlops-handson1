from __future__ import annotations

import json

import mlflow
from mlflow import MlflowClient

from src.common import MODEL_NAME, configure_mlflow, load_split
from src.compare_runs import SELECTION_PATH, best_candidate


def selected_model() -> dict:
    if SELECTION_PATH.exists():
        return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected, _ = best_candidate()
    return {
        "run_id": selected["run_id"],
        "run_name": selected["tags.mlflow.runName"],
        "model_uri": selected["tags.model_uri"],
        "val_auc": float(selected["metrics.val_auc"]),
    }


def main() -> None:
    configure_mlflow()
    selected = selected_model()
    model_version = mlflow.register_model(selected["model_uri"], MODEL_NAME)
    client = MlflowClient()
    client.update_model_version(
        MODEL_NAME,
        model_version.version,
        description=(
            f"Kandidat {selected['run_name']} dipilih berdasarkan val_auc="
            f"{selected['val_auc']:.6f}."
        ),
    )
    client.set_model_version_tag(MODEL_NAME, model_version.version, "validation_status", "passed")
    client.set_registered_model_alias(MODEL_NAME, "challenger", model_version.version)
    client.set_registered_model_alias(MODEL_NAME, "champion", model_version.version)

    _, X_val, _, _ = load_split()
    champion_uri = f"models:/{MODEL_NAME}@champion"
    champion = mlflow.pyfunc.load_model(champion_uri)
    predictions = champion.predict(X_val.head(3))
    print(f"Model {MODEL_NAME} v{model_version.version} -> @challenger dan @champion")
    print(f"URI: {champion_uri}")
    print(f"Prediksi contoh: {predictions.tolist()}")


if __name__ == "__main__":
    main()

