import os, time
from fastapi import FastAPI, HTTPException, Header, Response
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "demo-classifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

app = FastAPI(title="Model Serving")

# Prometheus Metrics
REQS = Counter("http_requests_total", "Total requests", ["path", "method", "status"])
LAT = Histogram("http_request_duration_seconds", "Request latency", ["path"], buckets=(0.05,0.1,0.2,0.5,1,2,5))
PREDICTIONS = Counter("model_predictions_total", "Total predictions made", ["model_name", "model_stage"])
PREDICTION_LATENCY = Histogram("model_prediction_duration_seconds", "Model prediction latency", buckets=(0.01,0.05,0.1,0.2,0.5,1,2))
MODEL_LOAD_TIME = Gauge("model_load_timestamp", "Timestamp when model was last loaded")
MODEL_INFO = Gauge("model_info", "Model information", ["model_name", "model_stage", "model_version"])
ERRORS = Counter("model_errors_total", "Total model errors", ["error_type"])

class PredictIn(BaseModel):
    features: list[list[float]]

_model = None

def load_model():
    global _model
    try:
        _model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        
        # Set model info metrics
        MODEL_LOAD_TIME.set(time.time())
        MODEL_INFO.labels(
            model_name=MODEL_NAME, 
            model_stage=MODEL_STAGE, 
            model_version="latest"
        ).set(1)
        
        print(f"Model loaded successfully: {MODEL_NAME}/{MODEL_STAGE}")
    except Exception as e:
        ERRORS.labels(error_type="model_load_error").inc()
        print(f"Error loading model: {e}")
        raise

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/health")
async def health():
    start_time = time.time()
    try:
        # Basic health check - verify model is loaded
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        status = "200"
        return {"status": "ok", "model": f"{MODEL_NAME}/{MODEL_STAGE}"}
    except HTTPException:
        status = "503"
        raise
    finally:
        # Record health check metrics
        total_time = time.time() - start_time
        LAT.labels(path="/health").observe(total_time)
        REQS.labels(path="/health", method="GET", status=status).inc()

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(body: PredictIn, x_api_key: str | None = Header(default=None)):
    start_time = time.time()
    status = "200"
    
    try:
        # Authentication check
        if API_KEY and x_api_key != API_KEY:
            status = "401"
            ERRORS.labels(error_type="authentication_error").inc()
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Convert list of lists to pandas DataFrame with proper column names
        # These are the wine dataset feature names
        feature_names = [
            'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
            'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
            'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(body.features, columns=feature_names)
        
        # Make prediction with timing
        prediction_start = time.time()
        preds = _model.predict(df)
        prediction_time = time.time() - prediction_start
        
        # Record prediction metrics
        PREDICTIONS.labels(model_name=MODEL_NAME, model_stage=MODEL_STAGE).inc()
        PREDICTION_LATENCY.observe(prediction_time)
        
        return {"predictions": preds.tolist()}
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 401)
        raise
    except Exception as e:
        status = "500"
        ERRORS.labels(error_type="prediction_error").inc()
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Record request metrics
        total_time = time.time() - start_time
        LAT.labels(path="/predict").observe(total_time)
        REQS.labels(path="/predict", method="POST", status=status).inc()