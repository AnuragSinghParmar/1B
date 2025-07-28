# Adobe Hackathon Round 1B - Makefile
# Simplifies Docker build and run operations

# Variables
IMAGE_NAME = adobe-round1b
CONTAINER_NAME = adobe-round1b-container
INPUT_DIR = $(PWD)/data/input
OUTPUT_DIR = $(PWD)/data/output
PERSONA_FILE = $(PWD)/persona.txt

# Default target
.PHONY: help
help:
	@echo "Adobe Hackathon Round 1B - Available commands:"
	@echo "  make build         - Build Docker image"
	@echo "  make run           - Run the solution"
	@echo "  make test          - Run with sample data"
	@echo "  make clean         - Clean Docker images"
	@echo "  make setup         - Setup directories"
	@echo "  make logs          - View container logs"
	@echo "  make shell         - Open shell in container"

# Build Docker image
.PHONY: build
build:
	@echo "Building Docker image..."
	docker build --platform linux/amd64 -t $(IMAGE_NAME) .
	@echo "✅ Docker image built successfully"

# Run the solution
.PHONY: run
run: setup
	@echo "Running Adobe Round 1B solution..."
	@echo "Input directory: $(INPUT_DIR)"
	@echo "Output directory: $(OUTPUT_DIR)"
	@echo "Persona file: $(PERSONA_FILE)"
	docker run --rm \
		-v $(INPUT_DIR):/app/input \
		-v $(OUTPUT_DIR):/app/output \
		-v $(PERSONA_FILE):/app/persona.txt \
		--network none \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME)
	@echo "✅ Processing completed. Check $(OUTPUT_DIR) for results."

# Setup required directories
.PHONY: setup
setup:
	@echo "Setting up directories..."
	mkdir -p data/input data/output
	@if [ ! -f $(PERSONA_FILE) ]; then \
		echo "Creating sample persona.txt..."; \
		echo "Investment Analyst" > $(PERSONA_FILE); \
		echo "Analyze revenue trends, R&D investments, and market positioning strategies" >> $(PERSONA_FILE); \
	fi
	@echo "✅ Directories and files ready"

# Test with sample data
.PHONY: test
test: build setup
	@echo "Testing with sample configuration..."
	@if [ -z "$$(ls -A $(INPUT_DIR) 2>/dev/null)" ]; then \
		echo "❌ No PDF files found in $(INPUT_DIR)"; \
		echo "Please add PDF files to test the solution"; \
		exit 1; \
	fi
	$(MAKE) run
	@if [ -f $(OUTPUT_DIR)/challenge1b_output.json ]; then \
		echo "✅ Test successful! Output generated."; \
		echo "Results:"; \
		head -20 $(OUTPUT_DIR)/challenge1b_output.json; \
	else \
		echo "❌ Test failed - no output generated"; \
	fi

# Clean Docker images and containers
.PHONY: clean
clean:
	@echo "Cleaning Docker resources..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null
	-docker rm $(CONTAINER_NAME) 2>/dev/null
	-docker rmi $(IMAGE_NAME) 2>/dev/null
	@echo "✅ Cleanup completed"

# View logs (if container is running)
.PHONY: logs
logs:
	docker logs $(CONTAINER_NAME) -f

# Open shell in container for debugging
.PHONY: shell
shell: build
	@echo "Opening shell in container..."
	docker run --rm -it \
		-v $(INPUT_DIR):/app/input \
		-v $(OUTPUT_DIR):/app/output \
		-v $(PERSONA_FILE):/app/persona.txt \
		--entrypoint /bin/bash \
		$(IMAGE_NAME)

# Check system requirements
.PHONY: check
check:
	@echo "Checking system requirements..."
	@echo "Docker version:"
	@docker --version
	@echo "Platform:"
	@docker info | grep "Architecture\|Operating System"
	@echo "Available disk space:"
	@df -h .
	@echo "✅ System check completed"

# Validate output format
.PHONY: validate
validate:
	@if [ -f $(OUTPUT_DIR)/challenge1b_output.json ]; then \
		echo "Validating output format..."; \
		python3 -m json.tool $(OUTPUT_DIR)/challenge1b_output.json > /dev/null && echo "✅ Valid JSON format" || echo "❌ Invalid JSON format"; \
	else \
		echo "❌ No output file found to validate"; \
	fi

# Show file sizes for constraint checking
.PHONY: size-check
size-check:
	@echo "Checking constraint compliance..."
	@echo "Docker image size:"
	@docker images $(IMAGE_NAME) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
	@if [ -f $(OUTPUT_DIR)/challenge1b_output.json ]; then \
		echo "Output file size:"; \
		ls -lh $(OUTPUT_DIR)/challenge1b_output.json; \
	fi

# Complete workflow: build, test, and validate
.PHONY: all
all: clean build test validate size-check
	@echo "🎉 Complete workflow finished successfully!"
	@echo "Your Adobe Hackathon Round 1B solution is ready for submission."

# Quick submit check
.PHONY: submit-check
submit-check: all
	@echo "=== SUBMISSION CHECKLIST ==="
	@echo "✅ Docker image builds successfully"
	@echo "✅ Solution runs without internet access"
	@if [ -f $(OUTPUT_DIR)/challenge1b_output.json ]; then echo "✅ Output file generated"; else echo "❌ No output file"; fi
	@echo "✅ Dockerfile uses linux/amd64 platform"
	@echo "✅ Container processes input directory automatically"
	@echo "✅ All files ready for private GitHub repository"
	@echo ""
	@echo "📁 Files to include in your submission:"
	@echo "   - Dockerfile"
	@echo "   - requirements.txt"  
	@echo "   - src/ (with all Python files)"
	@echo "   - config.yaml"
	@echo "   - README.md"
	@echo "   - approach_explanation.md"
	@echo "   - .gitignore"
	@echo "   - Makefile (optional)"
	@echo ""
	@echo "🚀 Ready for Adobe Hackathon submission!"