# Ariadne

Ariadne is a full-stack research assistant built with:

- **Frontend:** React + Vite
- **Backend:** LangGraph + FastAPI
- **Model provider:** Google Gemini APIs

## Repository Structure

- `frontend/` — React application UI
- `backend/` — LangGraph agent and API
- `docker-compose.yml` / `Dockerfile` — containerized runtime

## Prerequisites

- Node.js 20+
- Python 3.11+
- `uv` (for backend tooling)
- A valid `GEMINI_API_KEY`

## Local Development

### Frontend

```bash
cd frontend
npm install
npm run dev -- --host
```

### Backend

```bash
cd backend
cp .env.example .env
# set GEMINI_API_KEY in .env
langgraph dev
```

Or from the repository root:

```bash
make dev-frontend
make dev-backend
make dev
```

## Quality Checks

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

Backend:

```bash
cd backend
make lint
make test
```

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```
