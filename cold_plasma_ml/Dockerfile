FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g., for building some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure the local mlruns directory exists
RUN mkdir -p /app/mlruns

# Set PYTHONPATH to include the /app directory
ENV PYTHONPATH=/app

# Default command can be overridden
CMD ["python", "-m", "src.main"]
