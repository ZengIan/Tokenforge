# ===== Stage 1: Build frontend =====
FROM docker.1ms.run/library/node:20-slim AS frontend-builder

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ===== Stage 2: Python backend + serve static =====
FROM docker.1ms.run/library/python:3.11-slim

WORKDIR /app

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/app ./app
COPY backend/data ./data

# Copy built frontend into backend/dist
COPY --from=frontend-builder /build/dist ./dist

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
