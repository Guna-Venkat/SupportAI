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

# Copy the trained classifier model, retrieval index, and calibration/label encoders
COPY outputs/models/best_model/ ./outputs/models/best_model/
COPY outputs/models/label_encoder.json ./outputs/models/label_encoder.json
COPY outputs/metrics/calibration_summary.json ./outputs/metrics/calibration_summary.json
COPY outputs/retrieval_index/ ./outputs/retrieval_index/

# Upgrade pip and install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Expose default REST API port
EXPOSE 8000

# Run API server using the module runner to support dynamic PORT binding
CMD ["python", "-m", "src.api.app"]
