# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git build-essential curl file && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY ./requirements.txt /app/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install 'detectron2@git+https://github.com/facebookresearch/detectron2.git@main#egg=detectron2'

# Download model weights and config using a robust script
COPY ./download.py /app/download.py
RUN python /app/download.py

# Copy the application's code
COPY ./app /app/app

# Expose port and run the app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
