FROM python:3.10-slim

WORKDIR /app

# Install dependencies needed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Train the model only if model.pkl doesn't exist
RUN if [ ! -f "model/model.pkl" ]; then make train; fi

# Expose API port
EXPOSE 8000

# Create default env.js to rely on relative paths and Start API
CMD echo "window.RUNTIME_API_BASE = '${API_BASE:-}';" > static/env.js && uvicorn api.main:app --host 0.0.0.0 --port 8000
