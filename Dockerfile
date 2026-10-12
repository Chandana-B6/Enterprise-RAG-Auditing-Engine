# Use a lean, official Python runtime base image
FROM python:3.10-slim

# Set the working directory inside the container sandbox
WORKDIR /app

# Install system utilities needed for building C extensions (e.g. for numpy/chromadb)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to optimize Docker layer caching
COPY requirements.txt .

# Install all machine learning, vector DB, and UI packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the PyTorch embedding weights during build time.
# This prevents the container from being slow when a user first runs it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the modular application scripts into the workspace
COPY . .

# Expose Streamlit's standard operational network port
EXPOSE 8501

# Healthcheck flag to ensure the internal webserver is healthy
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to execute and launch the application on container boot
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
