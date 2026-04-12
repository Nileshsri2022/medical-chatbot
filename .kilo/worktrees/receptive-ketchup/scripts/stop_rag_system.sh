#!/bin/bash

# Medical RAG Chatbot - System Shutdown Script
# Healthcare

echo "🛑 Stopping Medical RAG Chatbot System"
echo "==============================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to kill process safely
kill_process() {
    local pid=$1
    local name=$2
    
    if [ ! -z "$pid" ] && kill -0 $pid 2>/dev/null; then
        print_status "Stopping $name (PID: $pid)..."
        kill $pid
        
        # Wait for graceful shutdown
        for i in {1..5}; do
            if ! kill -0 $pid 2>/dev/null; then
                print_status "$name stopped successfully"
                return 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        print_warning "Force killing $name..."
        kill -9 $pid 2>/dev/null
        sleep 1
        
        if ! kill -0 $pid 2>/dev/null; then
            print_status "$name force stopped"
        else
            print_error "Failed to stop $name"
        fi
    else
        print_warning "$name is not running"
    fi
}

# Function to kill processes by port
kill_by_port() {
    local port=$1
    local service_name=$2
    
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        print_status "Stopping $service_name on port $port..."
        echo "$pids" | xargs kill -9 2>/dev/null
        print_status "$service_name stopped"
    else
        print_status "No process found on port $port for $service_name"
    fi
}

# Read PIDs from file if exists
if [ -f "$SCRIPT_DIR/system_pids.txt" ]; then
    source "$SCRIPT_DIR/system_pids.txt"
    print_status "Found system PIDs file"
    
    # Stop services using PIDs
    kill_process "$RAG_PID" "RAG Backend Server"
    kill_process "$FRONTEND_PID" "Frontend Server"
    
    # Also kill by ports as backup
    if [ ! -z "$RAG_PORT" ]; then
        kill_by_port "$RAG_PORT" "RAG Backend"
    fi
    
    if [ ! -z "$FRONTEND_PORT" ]; then
        kill_by_port "$FRONTEND_PORT" "Frontend Server"
    fi
    
    # Clean up PID file
    rm -f "$SCRIPT_DIR/system_pids.txt"
    print_status "Cleaned up PID file"
else
    print_warning "No PID file found, killing by default ports..."
    
    # Kill by default ports
    kill_by_port "8001" "RAG Backend"
    kill_by_port "3000" "Frontend Server"
fi

# Clean up PID files in directories
print_status "Cleaning up PID files..."

if [ -f "$BACKEND_DIR/rag_server.pid" ]; then
    local rag_pid=$(cat "$BACKEND_DIR/rag_server.pid")
    kill_process "$rag_pid" "RAG Server (from PID file)"
    rm -f "$BACKEND_DIR/rag_server.pid"
fi

if [ -f "$FRONTEND_DIR/frontend_server.pid" ]; then
    local frontend_pid=$(cat "$FRONTEND_DIR/frontend_server.pid")
    kill_process "$frontend_pid" "Frontend Server (from PID file)"
    rm -f "$FRONTEND_DIR/frontend_server.pid"
fi

# Clean up temporary files
print_status "Cleaning up temporary files..."
rm -f "$FRONTEND_DIR/serve_frontend.py"
rm -f "$BACKEND_DIR/rag_server.log"
rm -f "$FRONTEND_DIR/frontend_server.log"

# Final verification
echo ""
print_status "Verifying shutdown..."

# Check ports
ports_to_check=("8001" "3000")
all_clear=true

for port in "${ports_to_check[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $port is still in use"
        all_clear=false
    else
        print_status "Port $port is clear"
    fi
done

if $all_clear; then
    echo ""
    echo "✅ Enhanced Medical RAG Chatbot System stopped successfully!"
    echo ""
    echo "🧹 Cleanup completed:"
    echo "   ✓ All processes terminated"
    echo "   ✓ Ports released"
    echo "   ✓ Temporary files removed"
    echo "   ✓ PID files cleaned"
    echo ""
    echo "To start the system again, run: ./scripts/start_rag_system.sh"
else
    echo ""
    print_warning "Some services may still be running. Check manually if needed."
    echo ""
    echo "Manual cleanup commands:"
    echo "   Kill by port: lsof -ti:8001 | xargs kill -9"
    echo "   Check processes: ps aux | grep -E '(rag_server|serve_frontend)'"
fi

echo ""
echo "System shutdown complete."