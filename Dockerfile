# Use an official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for MLflow UI
EXPOSE 5001

# Expose the port for Prefect server UI
EXPOSE 4200

# Run the Prefect and MLflow services
CMD ["prefect", "server", "start"]
