FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY train.py baseline_stats.py ./
ENTRYPOINT ["python", "train.py"]