# Use an official Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports (if needed)
EXPOSE 8000

# Start Supervisor
CMD ["supervisord", "-c", "/app/supervisord.conf"]
