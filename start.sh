#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

PORT=${PORT:-8000}

echo "=========================================="
echo " Starting AI Studio Phase 1 Backend Service "
echo "=========================================="

# Stop any running instances first
./stop.sh

# Ensure venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating venv..."
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
fi

# Check if redis-server is running on localhost:6379
if nc -z localhost 6379 2>/dev/null; then
    echo "🔴 Redis server detected on localhost:6379."
    echo "⚡ Launching Celery background worker..."
    PYTHONPATH=. ./venv/bin/celery -A app.workers.celery_worker worker --loglevel=info -P threads -c 4 > celery.log 2>&1 &
    CELERY_PID=$!
    echo "Celery worker running with PID: ${CELERY_PID} (Logs: tail -f celery.log)"
else
    echo "⚠️ Redis server is not running on localhost:6379."
    echo "   To enable Celery background tasks locally, start Redis using: redis-server"
fi

echo "Starting FastAPI server on http://localhost:${PORT}..."
PYTHONPATH=. ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --reload > server.log 2>&1 &

PID=$!
echo "Server started with PID: ${PID}"
echo "Server logs: tail -f server.log"

# Wait a moment and check health endpoint
sleep 2

if curl -s http://localhost:${PORT}/health > /dev/null; then
    echo "✅ AI Studio Backend is UP & RUNNING!"
    echo "📖 OpenAPI Documentation: http://localhost:${PORT}/docs"
else
    echo "⚠️ Server launching... Check server.log for details."
fi
