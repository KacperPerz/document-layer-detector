.PHONY: build run stop logs shell

# Variables
IMAGE_NAME := document-layer-detector
CONTAINER_NAME := document-layer-detector-container
PORT := 8000

# Commands
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

run:
	@echo "Running Docker container..."
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):$(PORT) $(IMAGE_NAME)

stop:
	@echo "Stopping Docker container..."
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

logs:
	@echo "Showing container logs..."
	docker logs -f $(CONTAINER_NAME)

shell:
	@echo "Accessing container shell..."
	docker exec -it $(CONTAINER_NAME) /bin/sh

# Default command
all: build
