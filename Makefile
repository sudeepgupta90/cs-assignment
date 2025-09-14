#    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
REGISTRY=mlops
VERSION?=0.1.0

# Images
build-images: build-trainer build-serving build-batch build-airflow build-mlflow

build-trainer:
	docker build -t $(REGISTRY)/trainer:$(VERSION) -f training/Dockerfile training
	minikube image load mlops/trainer:$(VERSION)

build-serving:
	docker build -t $(REGISTRY)/serving:$(VERSION) -f serving/Dockerfile serving
	minikube image load mlops/serving:$(VERSION)

build-batch:
	docker build -t $(REGISTRY)/batch:$(VERSION) -f batch_monitoring/Dockerfile batch_monitoring
	minikube image load mlops/batch:$(VERSION)


build-airflow:
	docker build -t $(REGISTRY)/airflow:$(VERSION) -f airflow/Dockerfile airflow
	minikube image load mlops/airflow:$(VERSION)


build-mlflow:
	docker build -t $(REGISTRY)/mlflow:$(VERSION) -f mlflow/Dockerfile mlflow
	minikube image load mlops/mlflow:$(VERSION)

# Deploy core data plane
deploy-core:
	kubectl apply -f k8s/namespace/namespace.yaml
	kubectl apply -f k8s/secrets/secrets.yaml
	kubectl apply -f k8s/mlops/minio.yaml
	kubectl apply -f k8s/mlops/mlflow.yaml
	kubectl apply -f k8s/mlops/postgres.yaml

# Deploy RBAC for Airflow cross-namespace pod creation
# deploy-rbac:
# 	kubectl apply -f k8s/airflow/rbac.yaml

# monitoring:
# 	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
# 	helm repo update
# 	helm upgrade --install kube-prom prometheus-community/kube-prometheus-stack \
# 	  --namespace monitoring --create-namespace \
# 	  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
# 	helm upgrade --install pushgateway prometheus-community/prometheus-pushgateway \
# 	  --namespace monitoring
# 	# kubectl apply -f k8s/monitoring/servicemonitors.yaml
# 	# kubectl apply -f k8s/monitoring/alerts.yaml

# Airflow
# airflow:
# 	helm repo add apache-airflow https://airflow.apache.org
# 	helm repo update
# 	helm upgrade --install airflow apache-airflow/airflow \
# 	  --namespace airflow --create-namespace \
# 	  --set images.airflow.repository=$(REGISTRY)/airflow \
# 	  --set images.airflow.tag=$(VERSION) \
# 	  --set webserver.defaultUser.username=$(AIRFLOW_USER) \
# 	  --set webserver.defaultUser.password=$(AIRFLOW_PASSWORD)

# App/Model plane
deploy-serving:
	kubectl apply -f k8s/serving/service.yaml
	kubectl apply -f k8s/serving/model-dev.yaml
	kubectl apply -f k8s/serving/model-prd.yaml

# Deploy the drift detection
batch:
	kubectl apply -f k8s/drift_detection/job.yaml

