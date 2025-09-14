# Walkthrough Steps

### Step 1: build-images
    
```
make build-images
```

### Step 2: deploy core plane

```
make deploy-core
```

### Step 3: create minio bucket

```
kubectl -n mlops port-forward svc/minio 9001:9001
```

open http://localhost:9001 → login (minioadmin/minioadmin) → create bucket: mlflow-artifacts

### Step 4: install prom

```
cd charts/observability/kube-prom
helm dependency update
helm upgrade --install kube-prom . --namespace monitoring -f values.yaml
```
	  
### Step 5: install push gateway

```
cd charts/observability/pushgateway
helm dependency update
helm upgrade --install pushgateway . -n monitoring
```

## Step 6: setup Service Monitors

```
kubectl apply -f airflow_sm.yaml 
kubectl apply -f model_serving_sm.yaml
kubectl apply -f drift.yaml  
```

### Step 7: Deploy Serving plane

```
make deploy-serving
```

At this point, you will notice that both model-serving-{dev,prod} are erroring out in mlops namespace. This is by design.


### Step-8: Train and put a model in staging(aka dev)

```
kubectl apply -f train.yaml
```

The model-serving-dev deployment will become stable in a few minutes, or you can always force a deployment restart

You will now notice that prod continues to be in CrashLoopBackOff.

### Step-9: Deploy model monitoring for drifts

```
make batch
```

### Step-10: Perform an API call to the dev model so deployed

Fwd the service from k8s to local
```
kubectl -n mlops port-forward deploy/model-serving-dev 8000:8000
```

Send this curl request
```
curl -sS -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: changeme-supersecret' \
  -d '{
    "features": [
      [13.2,2.3,2.5,16.8,98,2.1,1.8,0.5,1.6,4.8,1.0,3.0,750],
      [12.8,1.9,2.2,19.5,100,2.0,1.7,0.4,1.5,5.0,1.1,3.1,700]
    ]
  }'
```
You will see expected response: `{"predictions":[0,1]}%`

### Step-11: Trigger Drift Detection in K8s mlops ns
The job will trigger and push to metrics to prometheus via pushgateway

### Step-12: Deploy Airflow

```
cd charts/airflow
helm dependency update
helm upgrade --install airflow . -n airflow -f values.yaml --set airflowPodAnnotations.random=r$(uuidgen)
```

### Step-13: (Bonus) Trigger the prod dag

The prod dag will run and promote the model to production in mlflow which in turn will trigger the model serving prod deployment in mlops namespace to complete the cycle. This is not working as expected, there's some error which needs to be fixed which I could not track succesfully.

In order to make this work we need to enable multi namespace in Airflow, and additionally give Role + RoleBinding to Airflow SA's. This additional role is defined in the file at 
`k8s/airflow/rbac.yaml`