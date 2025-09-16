.PHONY: build run stop logs shell

# Variables
IMAGE_NAME := document-layer-detector
CONTAINER_NAME := document-layer-detector-container
PORT := 8000
API := http://localhost:$(PORT)
EVAL_IMG ?= app/data/example_data.png
EVAL_ANN ?= app/data/example_coco.json
IOU ?= 0.5
DETECT_IMG ?= $(EVAL_IMG)

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

# ------- Convenience test targets -------
.PHONY: wait detect-json detect-image eval-json-coco eval-image-coco eval-json-simple eval-image-simple evaluate detect

wait:
	@echo "Waiting for API at $(API)..."
	@until curl -sSf $(API)/health >/dev/null 2>&1; do \
	  sleep 1; \
	done; \
	echo "API is up"

detect-json: wait
	@echo "POST /detect (json)"
	curl -sS -X POST "$(API)/detect/?format=json" \
	  -H "accept: application/json" -H "Content-Type: multipart/form-data" \
	  -F "file=@app/data/example_data.png" | jq .

detect-image: wait
	@echo "POST /detect (image -> detections.png)"
	curl -sS -X POST "$(API)/detect/?format=image" \
	  -H "accept: image/png" -H "Content-Type: multipart/form-data" \
	  -F "file=@app/data/example_data.png" \
	  -o detections.png && file detections.png

eval-json-coco: wait
	@echo "POST /evaluate (json metrics, COCO)"
	curl -sS -X POST "$(API)/evaluate/?format=json&iou_threshold=0.5" \
	  -H "accept: application/json" -H "Content-Type: multipart/form-data" \
	  -F "file=@app/data/example_data.png" \
	  -F "annotations=@app/data/example_coco.json;type=application/json" | jq .

eval-image-coco: wait
	@echo "POST /evaluate (image -> compare.png, COCO)"
	curl -sS -X POST "$(API)/evaluate/?format=image&iou_threshold=0.5" \
	  -H "accept: image/png" -H "Content-Type: multipart/form-data" \
	  -F "file=@app/data/example_data.png" \
	  -F "annotations=@app/data/example_coco.json;type=application/json" \
	  -o compare.png && file compare.png

eval-json-simple: wait
	@echo "POST /evaluate (json metrics, simple)"
	curl -sS -X POST "$(API)/evaluate/?format=json&iou_threshold=0.5" \
	  -H "accept: application/json" -H "Content-Type: multipart/form-data" \
	  -F "file=@app/data/example_data.png" \
	  -F "annotations=@app/data/example_json.json;type=application/json" | jq .

eval-image-simple: wait
	@echo "POST /evaluate (image -> compare.png, simple)"
	curl -sS -X POST "$(API)/evaluate/?format=image&iou_threshold=0.5" \
	  -H "accept: image/png" -H "Content-Type: multipart/form-data" \
	  -F "file=@app/data/example_data.png" \
	  -F "annotations=@app/data/example_json.json;type=application/json" \
	  -o compare.png && file compare.png

# One-shot: metrics + comparison image using defaults (override EVAL_IMG/EVAL_ANN/IOU)
evaluate: wait
	@echo "POST /evaluate (format=both) on $(EVAL_IMG) with annotations $(EVAL_ANN)"
	curl -sS -X POST "$(API)/evaluate/?format=both&iou_threshold=$(IOU)" \
	  -H "accept: application/json" -H "Content-Type: multipart/form-data" \
	  -F "file=@$(EVAL_IMG)" \
	  -F "annotations=@$(EVAL_ANN);type=application/json" -o out.json
	@echo "--- Metrics ---"
	@jq .metrics out.json
	@python3 -c "import json,base64; d=json.load(open('out.json')); b=d.get('image_base64');\
 open('compare.png','wb').write(base64.b64decode(b)) if b else print('No image_base64 in response')" && \
	echo "Saved compare.png" || true
	@file compare.png 2>/dev/null || true

# One-shot detect with custom image (override DETECT_IMG)
detect: wait
	@echo "POST /detect (format=image) on $(DETECT_IMG) -> detections.png"
	curl -sS -X POST "$(API)/detect/?format=image" \
	  -H "accept: image/png" -H "Content-Type: multipart/form-data" \
	  -F "file=@$(DETECT_IMG)" -o detections.png && file detections.png
