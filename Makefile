.PHONY: help dev-frontend dev-backend dev

help:
	@echo "Available commands:"
	@echo "  make dev-frontend    - Starts the frontend development server (Vite)"
	@echo "  make dev-backend     - Starts the backend development server (Uvicorn with reload)"
	@echo "  make dev             - Starts both frontend and backend development servers"

dev-frontend:
	@echo "Starting frontend development server..."
	@cd frontend && npm run dev -- --host

dev-backend:
	@echo "Starting backend development server..."
	@cd backend && \
	if command -v langgraph >/dev/null 2>&1; then \
		langgraph dev; \
	elif python -c "import langgraph_cli" >/dev/null 2>&1; then \
		python -m langgraph_cli.cli dev; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run --with-editable . langgraph dev; \
	else \
		echo "Error: LangGraph CLI is not available. Install backend dependencies (e.g. 'cd backend && pip install .')." >&2; \
		exit 127; \
	fi

# Run frontend and backend concurrently
dev:
	@echo "Starting both frontend and backend development servers..."
	@make dev-frontend & make dev-backend 
