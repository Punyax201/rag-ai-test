# Use lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional, useful for some Python packages)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Command to run app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
