FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY environment.py .
COPY app.py .
COPY openenv.yaml .
COPY inference.py .
COPY server/ ./server/

EXPOSE 7860

CMD ["python", "app.py"]
