# Use official Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/model/saved_models/handwritten_digit_model.keras

# Copy requirements file first for optimal Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend application code and saved model artifact
COPY backend /app/backend
COPY model/saved_models /app/model/saved_models

# Expose backend port
EXPOSE 8000

# Set working directory to backend application module
WORKDIR /app/backend

# Run uvicorn server listening on 0.0.0.0 to accept external requests
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
