# Start from a small official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies first (this layer is cached for faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service code and the model artifact into the image
COPY ../service/ ./service/

# Document the port the service listens on
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
