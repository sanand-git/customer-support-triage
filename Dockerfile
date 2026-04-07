FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Start FastAPI server
CMD ["python", "app.py"]
