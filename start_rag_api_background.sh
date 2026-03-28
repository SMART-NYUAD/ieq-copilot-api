#!/bin/bash
# Start RAG API server in background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing RAG API processes
pkill -f "rag_api_server.py" 2>/dev/null
sleep 1

PORT=${1:-8001}
HOST=${2:-0.0.0.0}

echo "Starting RAG API server on $HOST:$PORT..."
nohup python3 rag_api_server.py $PORT $HOST > rag_api.log 2>&1 &

# Get PID
PID=$!
echo $PID > rag_api.pid
echo "RAG API started with PID: $PID"
echo "Logs: tail -f rag_api.log"
echo "Stop: ./stop_rag_api_background.sh"
echo ""
echo "API will be available at:"
echo "  http://localhost:$PORT"
echo "  http://localhost:$PORT/docs"
