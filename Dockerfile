FROM python:3.11.9-slim

WORKDIR /app

RUN mkdir -p logs

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
curl
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py .
COPY engine/ engine/

ENV FLASK_APP=run.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001
ENV HF_HOME=/model_cache

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "run:app", "--timeout", "300", "--log-file=/app/logs/gunicorn.log"]


