# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt

COPY requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

COPY data-prepare /opt/data-prepare
COPY mlops        /opt/mlops

EXPOSE 8000
CMD ["bash","-lc","uvicorn mlops.src.webapp:app --host 0.0.0.0 --port 8000"]

