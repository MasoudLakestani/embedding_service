FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY app.py ./
COPY functions/ ./functions/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir sentence-transformers>=5.2.0,<6.0.0 && \
    pip install --no-cache-dir fastapi>=0.115.0,<1.0.0 && \
    pip install --no-cache-dir "uvicorn[standard]>=0.32.0,<1.0.0"

# Expose port
EXPOSE 8585

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8585"]
