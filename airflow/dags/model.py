from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import os
import mlflow
from mlflow.tracking import MlflowClient


def check_model_metrics_and_promote(**context):
    """
    Check model metrics and promote to Production if threshold is met.
    Uses modern MLflow API to avoid deprecation warnings.
    """
    try:
        # Get configuration from Airflow Variables or environment
        model_name = "demo-classifier"
        accuracy_threshold = 0.85
        mlflow_tracking_uri = "http://mlflow.mlops.svc.cluster.local:5000"
        
        print(f"Checking metrics for model: {model_name}")
        print(f"Accuracy threshold: {accuracy_threshold}")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Initialize MLflow client
        client = MlflowClient()
        
        # Get latest Staging model version using modern API
        try:
            model_versions = client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                raise ValueError(f"No model versions found for model: {model_name}")
            
            # Find Staging versions
            staging_versions = [v for v in model_versions if v.current_stage == "Staging"]
            
            if not staging_versions:
                print("No models in Staging stage found")
                return {"status": "no_staging_models", "message": "No models in Staging stage"}
            
            # Get the latest Staging version
            latest_staging = max(staging_versions, key=lambda x: x.version)
            print(f"Found latest Staging model: version {latest_staging.version}")
            
        except Exception as e:
            print(f"Error getting model versions: {e}")
            raise
        
        # Get run metrics
        try:
            run = client.get_run(latest_staging.run_id)
            metrics = run.data.metrics
            
            print(f"Model metrics: {metrics}")
            
            accuracy = metrics.get('accuracy', 0.0)
            f1_score = metrics.get('f1', 0.0)
            
            print(f"Model accuracy: {accuracy}")
            print(f"Model F1 score: {f1_score}")
            
        except Exception as e:
            print(f"Error getting run metrics: {e}")
            raise
        
        # Check if model meets promotion criteria
        if accuracy >= accuracy_threshold:
            try:
                # Promote to Production
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_staging.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                
                message = f"Promoted model version {latest_staging.version} to Production (accuracy: {accuracy:.3f})"
                print(message)
                
                return {
                    "status": "promoted",
                    "version": latest_staging.version,
                    "accuracy": accuracy,
                    "message": message
                }
                
            except Exception as e:
                print(f"Error promoting model: {e}")
                raise
        else:
            message = f"Model version {latest_staging.version} stayed in Staging (accuracy: {accuracy:.3f} < {accuracy_threshold})"
            print(message)
            
            return {
                "status": "not_promoted",
                "version": latest_staging.version,
                "accuracy": accuracy,
                "message": message
            }
            
    except Exception as e:
        error_msg = f"Error in model promotion check: {e}"
        print(error_msg)
        raise Exception(error_msg)


def validate_training_completion(**context):
    """
    Validate that training completed successfully by checking MLflow for recent runs.
    """
    try:
        model_name = "demo-classifier"
        mlflow_tracking_uri = "http://mlflow.mlops.svc.cluster.local:5000"
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient()
        
        # Get recent runs for the model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            raise ValueError(f"No model versions found for model: {model_name}")
        
        # Get the latest version
        latest_version = max(model_versions, key=lambda x: x.version)
        
        # Check if the run is recent (within last hour)
        run = client.get_run(latest_version.run_id)
        run_time = datetime.fromtimestamp(run.info.start_time / 1000)
        current_time = datetime.now()
        
        if (current_time - run_time).total_seconds() > 3600:  # 1 hour
            raise ValueError(f"Latest run is too old: {run_time}")
        
        print(f"Training validation successful. Latest run: {latest_version.version}")
        return {"status": "validated", "version": latest_version.version}
        
    except Exception as e:
        error_msg = f"Training validation failed: {e}"
        print(error_msg)
        raise Exception(error_msg)


# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

# Create the DAG
with DAG(
    dag_id="mlops_model_pipeline",
    default_args=default_args,
    description="MLOps Model Training, Validation, and Deployment Pipeline",
    schedule="@daily",  # Run daily
    max_active_runs=1,  # Prevent concurrent runs
    tags=["mlops", "model-training", "deployment"],
) as dag:

    # Task 1: Run model training
    train_model = KubernetesPodOperator(
        task_id="train_model",
        name="trainer-pod",
        namespace="airflow",
        image="mlops/trainer:0.1.0",
        cmds=["python"],
        arguments=["/app/train.py"],
        env_vars={
            "MODEL_NAME": "demo-classifier",
            "MLFLOW_TRACKING_URI": "http://mlflow.mlops.svc.cluster.local:5000",
            "AWS_DEFAULT_REGION": "us-east-1"
        },
        secrets=[
            {
                "secret_name": "mlops-secrets",
                "key": "AWS_ACCESS_KEY_ID",
                "env_var": "AWS_ACCESS_KEY_ID"
            },
            {
                "secret_name": "mlops-secrets", 
                "key": "AWS_SECRET_ACCESS_KEY",
                "env_var": "AWS_SECRET_ACCESS_KEY"
            },
            {
                "secret_name": "mlops-secrets",
                "key": "API_KEY", 
                "env_var": "API_KEY"
            }
        ],
        
        get_logs=True,
        is_delete_operator_pod=False,
        retries=1,
        retry_delay=timedelta(minutes=2),
        container_resources={
            "requests": {
                "memory": "512Mi",
                "cpu": "200m"
            },
            "limits": {
                "memory": "1Gi",
                "cpu": "500m"
            }
        }
    )

    # Task 2: Validate training completion
    validate_training = PythonOperator(
        task_id="validate_training",
        python_callable=validate_training_completion,
        retries=1,
        retry_delay=timedelta(minutes=2),
    )

    # Task 3: Check metrics and promote model
    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=check_model_metrics_and_promote,
        retries=1,
        retry_delay=timedelta(minutes=2),
    )

    # Define task dependencies
    train_model >> validate_training >> promote_model