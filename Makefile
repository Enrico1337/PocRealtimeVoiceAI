.PHONY: help build up down logs clean dev-up dev-down health shell-orchestrator shell-tts

# Default target
help:
	@echo "POC Realtime Voice AI - Makefile Commands"
	@echo "==========================================="
	@echo ""
	@echo "Production (GPU):"
	@echo "  make build     - Build all containers"
	@echo "  make up        - Start all services with GPU support"
	@echo "  make down      - Stop all services"
	@echo "  make logs      - Follow logs from all services"
	@echo "  make health    - Check health status of all services"
	@echo ""
	@echo "Development (CPU only):"
	@echo "  make dev-up    - Start services without GPU (for testing)"
	@echo "  make dev-down  - Stop dev services"
	@echo ""
	@echo "Utilities:"
	@echo "  make shell-orchestrator  - Open shell in orchestrator container"
	@echo "  make shell-tts           - Open shell in tts container"
	@echo "  make clean               - Remove all containers, volumes, and images"
	@echo ""

# =============================================================================
# Production Commands (GPU)
# =============================================================================

build:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml build

up:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

down:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml down

logs:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f

logs-orchestrator:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f orchestrator

logs-llm:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f llm

logs-stt:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f stt

logs-tts:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f tts

# =============================================================================
# Development Commands (CPU only - for testing builds)
# =============================================================================

dev-up:
	@echo "Starting in CPU-only mode (for build testing)..."
	docker compose -f docker-compose.yml up -d

dev-down:
	docker compose -f docker-compose.yml down

# =============================================================================
# Utility Commands
# =============================================================================

health:
	@echo "Checking service health..."
	@echo ""
	@echo "Orchestrator:"
	@curl -s http://localhost:7860/health || echo "Not reachable"
	@echo ""
	@echo ""
	@echo "Container status:"
	@docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps

shell-orchestrator:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec orchestrator /bin/bash

shell-tts:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec tts /bin/bash

clean:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml down -v --rmi local
	@echo "Cleaned up containers, volumes, and local images"

# =============================================================================
# Quick restart commands
# =============================================================================

restart-orchestrator:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart orchestrator

restart-tts:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart tts

rebuild-orchestrator:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml build orchestrator
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d orchestrator

rebuild-tts:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml build tts
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d tts
