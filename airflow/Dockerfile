FROM apache/airflow:3.0.6

# Install Python packages with specific versions for compatibility
RUN pip install mlflow==3.3.2 apache-airflow-providers-cncf-kubernetes==10.7.0 google-re2

COPY dags/ /opt/airflow/dags/
