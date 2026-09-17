# syntax=docker/dockerfile:1
# ============================================================
# Farmer Crop Advisory System — CPU Docker Image
# ============================================================
# Runs Streamlit on CPU. No GPU driver setup required.
# Image: ~3.5 GB (Python 3.11 + TensorFlow CPU + Streamlit)
# ============================================================

FROM python:3.11-slim

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Install Python dependencies before copying source
# (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        tensorflow==2.21.0 \
        numpy \
        pandas \
        scikit-learn \
        matplotlib \
        seaborn \
        pillow \
        streamlit

# Copy source code (excluding large data/models via .dockerignore)
COPY config.py main.py farm_advisor.py ./
COPY src/ ./src/

# Directories that will be mounted as volumes at runtime
RUN mkdir -p models data/raw data/processed outputs/figures outputs/reports outputs/predictions

# Change ownership
RUN chown -R appuser:appuser /app
USER appuser

# Streamlit default port
EXPOSE 8501

# Streamlit config: disable browser auto-open & telemetry
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
