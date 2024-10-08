FROM python:3.12.6-slim

# Install system dependencies, including graphviz and git
RUN apt-get update && apt-get install -y \
    graphviz \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for the backend code
WORKDIR /app/backend

# Copy the requirements file from the backend directory into the container
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source code into the container
COPY backend/ .

# Expose the backend port (8000 as per previous Dockerfile)
EXPOSE 8000

# Command to run the backend application
CMD ["python3", "main.py"]
