# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /repo

# Copy and install dependencies
COPY requirements.txt .
RUN sed -i 's/cu101/cpu/' requirements.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy relevant directories
COPY text_recognizer/ text_recognizer/
COPY api_server/ api_server/

# Expose port and set environment
EXPOSE 8000
ENV PYTHONPATH=/repo

# Default command
CMD ["python3", "api_server/app.py"]