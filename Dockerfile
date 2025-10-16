# Multi-stage build: Build Next.js frontend, then run Flask backend

# Stage 1: Build Next.js frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Flask backend
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Flask app files
COPY app.py background_remover_light.py ./

# Copy built Next.js frontend from builder stage
COPY --from=frontend-builder /app/frontend/out ./frontend/out

# Create files directory
RUN mkdir -p files

# Expose port
EXPOSE 5000

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]