# Start from an official, clean Python 3.11 environment
FROM python:3.11-slim

# Set an argument for the User ID, defaulting to a standard value
ARG UID=1000

# Set a working directory inside the container
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy our requirements file
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# --- THIS IS THE FINAL PERMISSION FIX ---
# Create a non-root user WITH THE SAME UID as the host user
RUN useradd --create-home --shell /bin/bash --uid ${UID} appuser

# Copy our project's source code
# We do this AFTER creating the user
COPY . .

# Grant ownership of the app directory to our new user
RUN chown -R appuser:appuser /app

# Switch to the new, unprivileged user
USER appuser

# Expose port 8001
EXPOSE 8001

# The command that will run when the container starts
CMD ["uvicorn", "ai_core.main:app", "--host", "0.0.0.0", "--port", "8001"]