import os
import json
import time
import mlflow
import pandas as pd
import numpy as np
import requests
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


def main():
    """Main function to run batch drift detection."""
    try:
        # Environment variables
        PUSH = os.getenv("PUSHGATEWAY_URL")
        MODEL_NAME = os.getenv("MODEL_NAME", "demo-classifier")
        STAGE = os.getenv("MODEL_STAGE", "Production")
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        
        if not MLFLOW_TRACKING_URI:
            raise ValueError("MLFLOW_TRACKING_URI environment variable is required")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        print(f"Starting batch drift detection for model: {MODEL_NAME}/{STAGE}")
        
        # Initialize MLflow client
        client = mlflow.MlflowClient()
        
        # Get model version using modern API (avoiding deprecated stages)
        try:
            # Try to get the latest version first
            model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            
            if not model_versions:
                raise ValueError(f"No model versions found for model: {MODEL_NAME}")
            
            # Filter by stage if specified
            if STAGE and STAGE != "None":
                staged_versions = [v for v in model_versions if v.current_stage == STAGE]
                if staged_versions:
                    # Get the latest staged version
                    latest_version = max(staged_versions, key=lambda x: x.version)
                else:
                    print(f"No versions found in stage '{STAGE}', using latest version")
                    latest_version = max(model_versions, key=lambda x: x.version)
            else:
                # Get the latest version regardless of stage
                latest_version = max(model_versions, key=lambda x: x.version)
            
            print(f"Using model version: {latest_version.version} (stage: {latest_version.current_stage})")
            
        except Exception as e:
            print(f"Error getting model version: {e}")
            raise
        
        # Get run ID and download baseline
        run_id = latest_version.run_id
        
        try:
            # Download baseline artifacts
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path="baseline/baseline.json"
            )
            baseline = pd.read_json(local_path)
            print(f"Loaded baseline statistics for {len(baseline.columns)} features")
            
        except Exception as e:
            print(f"Error downloading baseline: {e}")
            # Create a dummy baseline if artifact doesn't exist
            print("Creating dummy baseline for demo purposes")
            feature_names = [
                'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
            ]
            baseline_data = {
                'mean': np.random.uniform(0, 20, len(feature_names)),
                'std': np.random.uniform(0.1, 5, len(feature_names)),
                'min': np.random.uniform(0, 10, len(feature_names)),
                'max': np.random.uniform(10, 30, len(feature_names))
            }
            baseline = pd.DataFrame(baseline_data, index=feature_names)
        
        # Generate demo data (in practice, read from MinIO/S3)
        print("Generating demo data for drift detection...")
        n_samples = 256
        X = pd.DataFrame(
            np.random.normal(
                baseline.loc["mean"], 
                baseline.loc["std"], 
                size=(n_samples, len(baseline.columns))
            ), 
            columns=baseline.columns
        )
        
        # PSI calculation function
        def psi(expected, actual, buckets=10):
            """Calculate Population Stability Index between expected and actual distributions."""
            try:
                qs = np.linspace(0, 1, buckets + 1)
                cuts = np.unique(np.quantile(expected, qs))
                
                if len(cuts) < 2:
                    return 0.0
                
                e_hist = np.histogram(expected, bins=cuts)[0] / len(expected)
                a_hist = np.histogram(actual, bins=cuts)[0] / len(actual)
                
                # Avoid division by zero
                e_hist = np.where(e_hist == 0, 1e-6, e_hist)
                a_hist = np.where(a_hist == 0, 1e-6, a_hist)
                
                return np.sum((a_hist - e_hist) * np.log(a_hist / e_hist))
            except Exception as e:
                print(f"Error calculating PSI: {e}")
                return 0.0
        
        # Calculate PSI for each feature
        print("Calculating PSI for each feature...")
        psis = {}
        for feature in X.columns:
            try:
                # Generate expected distribution from baseline
                expected = baseline.loc["mean", feature] + np.random.randn(1000) * baseline.loc["std", feature]
                actual = X[feature]
                
                psi_value = psi(expected, actual)
                psis[feature] = psi_value
                print(f"PSI for {feature}: {psi_value:.4f}")
                
            except Exception as e:
                print(f"Error calculating PSI for {feature}: {e}")
                psis[feature] = 0.0
        
        # Calculate average PSI
        avg_psi = float(np.mean(list(psis.values())))
        print(f"Average PSI: {avg_psi:.4f}")
        
        # Push metrics to Pushgateway
        if PUSH:
            try:
                # Create Prometheus metrics
                registry = CollectorRegistry()
                psi_gauge = Gauge('batch_feature_drift_psi', 'Average PSI for batch drift detection', registry=registry)
                psi_gauge.set(avg_psi)
                
                # Push to gateway
                push_to_gateway(
                    gateway=PUSH,
                    job='batch_drift',
                    registry=registry,
                    timeout=5
                )
                print(f"Successfully pushed PSI metric: {avg_psi:.4f}")
                
            except Exception as e:
                print(f"Error pushing to Pushgateway: {e}")
                # Fallback to direct HTTP request
                try:
                    metric_body = f"batch_feature_drift_psi {avg_psi}\n"
                    response = requests.post(
                        f"{PUSH}/metrics/job/batch_drift", 
                        data=metric_body, 
                        timeout=5
                    )
                    response.raise_for_status()
                    print(f"Successfully pushed PSI metric via HTTP: {avg_psi:.4f}")
                except Exception as e2:
                    print(f"Error with HTTP fallback: {e2}")
        else:
            print("PUSHGATEWAY_URL not set, skipping metric push")
        
        print("Batch drift detection completed successfully")
        
    except Exception as e:
        print(f"Error in batch drift detection: {e}")
        raise


if __name__ == "__main__":
    main()