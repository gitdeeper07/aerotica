# AEROTICA Dockerfile
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    libnetcdf-dev \
    libhdf5-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .
COPY setup.cfg .
COPY MANIFEST.in .

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgdal30 \
    libnetcdf19 \
    libhdf5-103-1 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create aerotica user
RUN useradd -m -s /bin/bash aerotica

# Copy Python packages from builder
COPY --from=builder /root/.local /home/aerotica/.local

# Set environment variables
ENV PATH=/home/aerotica/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH \
    AEROTICA_CONFIG=/app/config/production.yaml \
    PYTHONUNBUFFERED=1

# Create directory structure
WORKDIR /app
RUN mkdir -p /app/src /app/data /app/models /app/config /app/logs /app/reports

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Copy model weights (optional, can be mounted as volume)
# COPY models/ ./models/

# Set ownership
RUN chown -R aerotica:aerotica /app

# Switch to aerotica user
USER aerotica

# Install aerotica package
RUN pip install --user -e .

# Expose ports
EXPOSE 8000 8501 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "aerotica.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands:
# CMD ["python", "scripts/launch_dashboard.py"]
# CMD ["python", "scripts/gust_prealert.py", "--config", "config/tokyo.yaml"]
