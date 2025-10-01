# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# All dependencies are now in requirements.txt

# Copy the entire project
COPY . .

# Expose port 8080
EXPOSE 10000

# Set the command to run the portfolio server
CMD ["python", "finance/agent/portfolio_server.py", "--host", "0.0.0.0", "--port", "10000"]

