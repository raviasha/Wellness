# --- Stage 1: Build Frontend ---
FROM node:20-slim AS builder
WORKDIR /app/webapp
COPY webapp/package.json webapp/package-lock.json ./
RUN npm install
COPY webapp/ ./
# Neutralize rewrites in next.config.ts if they rely on local env
RUN npm run build

# --- Stage 2: Final Image ---
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from Stage 1
COPY --from=builder /app/webapp/out ./static

# Ensure the models directory exists (even if empty)
RUN mkdir -p models

# Expose the HF Space port
EXPOSE 7860

# Start the application
CMD ["python", "app.py"]
