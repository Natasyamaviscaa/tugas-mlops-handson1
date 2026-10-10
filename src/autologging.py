from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import common_tags, configure_mlflow, load_split, log_run_context


def main() -> None:
    configure_mlflow()
    X_train, X_val, y_train, y_val = load_split()
    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=42))]
    )

    mlflow.sklearn.autolog(log_models=False, silent=False)
    try:
        with mlflow.start_run(run_name="logreg-autolog") as run:
            log_run_context(X_train, y_train)
            mlflow.set_tags(common_tags(stage="autolog", candidate=True))
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_val)[:, 1]
            predictions = model.predict(X_val)
            mlflow.log_metrics(
                {
                    "val_f1": f1_score(y_val, predictions),
                    "val_auc": roc_auc_score(y_val, probabilities),
                }
            )
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=infer_signature(X_train, model.predict(X_train)),
                input_example=X_val.head(3),
            )
            mlflow.set_tag("model_uri", model_info.model_uri)
    finally:
        mlflow.sklearn.autolog(disable=True)

    print(f"Run autolog tersimpan: {run.info.run_id}")


if __name__ == "__main__":
    main()

