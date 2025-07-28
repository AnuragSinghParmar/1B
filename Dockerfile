FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY src/ src/
COPY config.yaml .

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Set the default command
CMD ["python", "src/main.py", "--input_dir", "/app/input", "--output_dir", "/app/output", "--config", "config.yaml"]