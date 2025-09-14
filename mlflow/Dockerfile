FROM python:3.11-slim

RUN pip install mlflow boto3 psycopg2-binary gunicorn

COPY server.sh /server.sh

RUN chmod +x /server.sh

EXPOSE 5000

ENTRYPOINT ["/server.sh"]