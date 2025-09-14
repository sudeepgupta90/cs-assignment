# training/train.py
from __future__ import annotations
import os
import hashlib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from baseline_stats import compute_baseline_df, save_baseline_json

# ---------- Config via env ----------
EXPERIMENT   = os.getenv("EXPERIMENT", "demo-exp")
MODEL_NAME   = os.getenv("MODEL_NAME", "demo-classifier")
TEST_SIZE    = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "200"))
MAX_DEPTH    = int(os.getenv("MAX_DEPTH", "6"))

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.mlops:5000"))
mlflow.set_experiment(EXPERIMENT)

def _hash_df(df: pd.DataFrame) -> str:
    # Fast, order-sensitive hash of the dataset
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def main() -> None:
    X, y = load_wine(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    with mlflow.start_run() as run:
        # ---------- Train ----------
        params = {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "random_state": RANDOM_STATE,
        }
        clf = RandomForestClassifier(**params).fit(Xtr, ytr)
        preds = clf.predict(Xte)

        metrics = {
            "accuracy": accuracy_score(yte, preds),
            "f1_macro": f1_score(yte, preds, average="macro"),
        }

        # ---------- Baseline (separate module) ----------
        baseline_df = compute_baseline_df(Xtr)
        save_baseline_json(baseline_df, "baseline.json")

        # ---------- Log to MLflow ----------
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("data_hash_full", _hash_df(pd.concat([X, y], axis=1)))
        mlflow.log_param("data_hash_trainX", _hash_df(Xtr))
        mlflow.log_artifact("baseline.json", artifact_path="baseline")

        signature = infer_signature(Xtr, clf.predict(Xtr))
        input_example = Xtr.iloc[:5]

        mlflow.sklearn.log_model(
            clf,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
        )

        # ---------- Promote to Staging (deterministic version lookup) ----------
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name = '{MODEL_NAME}' and run_id = '{run.info.run_id}'"
        )
        if not versions:
            raise RuntimeError("Registered model version not found for this run.")
        # Choose the newest version for this run
        version = max(int(v.version) for v in versions)
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        print(f"Run {run.info.run_id} logged. Promoted {MODEL_NAME} v{version} → Staging.")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()
