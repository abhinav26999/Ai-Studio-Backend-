#!/usr/bin/env bash

PORT=${PORT:-8000}

echo "Stopping any server running on port ${PORT}, uvicorn instances, or Celery workers..."

# Find and kill process listening on port 8000
PIDS=$(lsof -ti:${PORT})

if [ -n "$PIDS" ]; then
    echo "Found running processes on port ${PORT}: ${PIDS}"
    kill -9 $PIDS 2>/dev/null
    echo "Stopped processes on port ${PORT}."
else
    echo "No process running on port ${PORT}."
fi

# Kill any remaining uvicorn processes for this app
pkill -f "uvicorn app.main:app" 2>/dev/null || true

# Kill any remaining Celery worker processes
pkill -9 -f "celery" 2>/dev/null || true

echo "AI Studio Backend and Celery workers stopped successfully."
