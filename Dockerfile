# Start from an official, clean Python 3.11 environment
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Install system-level dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy our requirements file into the container
COPY requirements.txt .

# --- THIS IS THE CORRECTED LINE ---
# Install Python packages using --extra-index-url to search both PyPI and the PyTorch index
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy our entire project's code into the container
COPY . .

# Make our server script executable
RUN chmod +x run_server.sh

# Expose port 8000 to the outside world
EXPOSE 8000

# The command that will run when the container starts
CMD ["./run_server.sh"]