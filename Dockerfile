# Use official Python image as a base
FROM python:3.11-slim

# Set environment variables to reduce Python buffer and logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py /app/
COPY generate_model.py /app/
COPY templates /app/templates

# Install Python dependencies
RUN pip install --no-cache-dir prefect flask pandas numpy yfinance scikit-learn tensorflow mlflow joblib

# Expose Flask app's port
EXPOSE 5005

# Default command to start Flask app
CMD ["python", "app.py"]
