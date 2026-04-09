FROM python:3.11-slim

# Cache bust: v6
WORKDIR /app

COPY requirements.txt .

# Install all packages, ignore openenv-core if not found on PyPI yet
RUN pip install --no-cache-dir fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3 openai==1.57.2 pyyaml==6.0.2 httpx==0.27.2 && \
    pip install --no-cache-dir openenv-core>=0.2.0 || echo "openenv-core not yet on PyPI, skipping"

COPY environment.py .
COPY app.py .
COPY openenv.yaml .
COPY inference.py .
COPY pyproject.toml .
COPY server/ ./server/

EXPOSE 7860

CMD ["python", "app.py"]
