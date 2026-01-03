FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and poetry
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Configure poetry
RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10 && \
    poetry config installer.no-binary :none:

# Copy project files
COPY pyproject.toml ./
COPY app.py ./
COPY functions/ ./functions/
COPY schemas/ ./schemas/

# Install PyTorch CPU-only first (to avoid CUDA dependencies)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies using poetry (with increased timeout)
RUN poetry install --no-root --no-interaction --no-ansi

# Expose port
EXPOSE 8585

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8585"]
