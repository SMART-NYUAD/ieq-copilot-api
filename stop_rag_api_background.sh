#!/bin/bash
# Stop RAG API server started in background from this directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f rag_api.pid ]; then
    PID=$(cat rag_api.pid)
    echo "Stopping RAG API (PID: $PID)..."
    kill $PID 2>/dev/null
    sleep 2

    # Force kill if still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "Force killing..."
        kill -9 $PID 2>/dev/null
    fi

    rm -f rag_api.pid
    echo "RAG API stopped"
else
    echo "No PID file found. Killing all RAG API processes..."
    pkill -f "rag_api_server.py"
    echo "Done"
fi

