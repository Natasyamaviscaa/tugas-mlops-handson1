from __future__ import annotations

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.common import common_tags, configure_mlflow, load_split, log_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning dengan nested MLflow runs")
    parser.add_argument("--trials", type=int, default=20, help="Jumlah trial Optuna")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials harus minimal 1")

    import mlflow
    import mlflow.sklearn
    import optuna
    from mlflow.models import infer_signature

    configure_mlflow()
    X_train, X_val, y_train, y_val = load_split()

    with mlflow.start_run(run_name="optuna-rf-sweep") as parent_run:
        log_run_context(X_train, y_train)
        mlflow.set_tags(common_tags(stage="tuning", candidate=True))
        search_space = {
            "n_estimators": {"low": 50, "high": 300, "step": 50},
            "max_depth": {"low": 3, "high": 12},
            "min_samples_split": {"low": 2, "high": 10},
        }
        mlflow.log_dict(search_space, "search_space.json")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "random_state": 42,
            }
            with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
                mlflow.set_tags(common_tags(stage="tuning-trial", candidate=False))
                mlflow.log_params(params)
                model = RandomForestClassifier(**params).fit(X_train, y_train)
                score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                mlflow.log_metric("val_auc", score)
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=args.trials)

        best_params = {**study.best_params, "random_state": 42}
        best_model = RandomForestClassifier(**best_params).fit(X_train, y_train)
        predictions = best_model.predict(X_val)
        mlflow.log_params({f"best_{key}": value for key, value in study.best_params.items()})
        mlflow.log_metrics(
            {
                "val_auc": study.best_value,
                "best_val_auc": study.best_value,
            }
        )
        mlflow.log_param("trial_count", args.trials)
        model_info = mlflow.sklearn.log_model(
            best_model,
            name="model",
            signature=infer_signature(X_train, best_model.predict(X_train)),
            input_example=X_val.head(3),
        )
        mlflow.set_tag("model_uri", model_info.model_uri)
        mlflow.log_metric("val_accuracy", (predictions == y_val).mean())

    print(f"Parent run: {parent_run.info.run_id}")
    print(f"Best val_auc={study.best_value:.6f}; params={study.best_params}")


if __name__ == "__main__":
    main()

