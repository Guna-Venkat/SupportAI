# Base python image
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (libgomp1 is required by FAISS on Linux CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging and configuration files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/
COPY configs/ ./configs/

# Upgrade pip and install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Expose REST API port
EXPOSE 8000

# Run API server with uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
