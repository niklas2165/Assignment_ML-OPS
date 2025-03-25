# Use an official Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install supervisor
RUN apt-get update && \
    apt-get install -y supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all files into container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["supervisord", "-c", "/app/supervisord.conf"]

EXPOSE 8000
