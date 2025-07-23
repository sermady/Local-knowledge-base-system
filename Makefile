# Kimi Knowledge Base - Development Makefile

.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  clean        - Clean up generated files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-up    - Start services with Docker Compose"
	@echo "  docker-down  - Stop services"
	@echo "  run          - Run the application locally"

# Installation
install:
	pip install -r requirements.txt

dev-install: install
	pip install pytest pytest-asyncio httpx black isort flake8 mypy

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Docker operations
docker-build:
	docker build -t kimi-knowledge-base .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Local development
run:
	python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Database operations
init-db:
	mkdir -p data/qdrant data/cache
	touch data/cache.db

# Monitoring (optional)
monitoring-up:
	docker-compose --profile monitoring up -d

monitoring-down:
	docker-compose --profile monitoring down