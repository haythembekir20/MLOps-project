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

# Install Python dependencies and Cron
RUN apt-get update && apt-get install -y cron && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add a cron job to run the script every 24 hours
RUN echo "0 0 * * * python /app/generate_model.py >> /var/log/cron.log 2>&1" > /etc/cron.d/generate_model && \
    chmod 0644 /etc/cron.d/generate_model && \
    crontab /etc/cron.d/generate_model

# Start Cron and Flask
CMD ["sh", "-c", "cron && python app.py"]
