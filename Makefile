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
	PY_CMD=; \
	if command -v python3 >/dev/null 2>&1; then \
		PY_CMD=python3; \
	elif command -v python >/dev/null 2>&1; then \
		PY_CMD=python; \
	fi; \
	if command -v langgraph >/dev/null 2>&1; then \
		langgraph dev; \
	elif [ -n "$$PY_CMD" ] && $$PY_CMD -c "import langgraph_cli.cli" >/dev/null 2>&1; then \
		$$PY_CMD -m langgraph_cli.cli dev; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run langgraph dev; \
	elif [ -n "$$PY_CMD" ]; then \
		echo "Error: LangGraph CLI is not available. From the backend directory, install dependencies (e.g. '$$PY_CMD -m pip install -e .'), or run with uv (e.g. 'uv run langgraph dev')." >&2; \
		exit 127; \
	else \
		echo "Error: LangGraph CLI is not available. Install backend dependencies (e.g. 'pip install -e .') or run with uv (e.g. 'uv run langgraph dev')." >&2; \
		exit 127; \
	fi

# Run frontend and backend concurrently
dev:
	@echo "Starting both frontend and backend development servers..."
	@make dev-frontend & make dev-backend 
