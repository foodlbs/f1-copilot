# F1 Race Strategy Analyzer
# Multi-stage Docker build for production deployment

# =============================================================================
# Stage 1: Build stage
# =============================================================================
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================================================================
# Stage 2: Production stage
# =============================================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="F1 Race Strategy Analyzer"
LABEL version="1.0.0"
LABEL description="AI-powered Formula 1 race strategy analysis system"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/cache /app/models /app/logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "fastapi_backend.py"]

# =============================================================================
# Stage 3: Development stage (optional)
# =============================================================================
FROM production as development

# Switch back to root for dev setup
USER root

# Install development tools
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    mypy \
    httpx

# Switch back to appuser
USER appuser

# Development command with auto-reload
CMD ["uvicorn", "fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
