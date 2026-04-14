#!/bin/bash

# Medical RAG Chatbot - Complete System Startup Script
# Healthcare - Advanced Conversational AI

echo "ðŸ¥ Medical RAG Chatbot System"
echo "==================================="
echo "ðŸ¥ Healthcare Organization"
echo "ðŸ§  Advanced Conversational AI with Memory & Context"
echo ""

# Configuration
RAG_PORT=8002
FRONTEND_PORT=3000
LLM_PORT=8001
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
PYTHON_CMD=""
FRONTEND_NO_CACHE="${FRONTEND_NO_CACHE:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Detect available Python command
set_python_command() {
    if [ -x "$PROJECT_ROOT/.venv/Scripts/python.exe" ]; then
        PYTHON_CMD="$PROJECT_ROOT/.venv/Scripts/python.exe"
    elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
    elif command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is required but not installed"
        exit 1
    fi

    print_status "âœ“ Python found: $($PYTHON_CMD --version 2>&1)"
}

# Function to check if port is in use (cross-platform)
check_port() {
    local port=$1
    if command -v lsof &>/dev/null; then
        lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1
    elif command -v netstat &>/dev/null; then
        netstat -an 2>/dev/null | grep -q ":${port}.*LISTEN"
    elif command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${port} "
    else
        # Fallback: try connecting to the port
        (echo >/dev/tcp/localhost/$port) 2>/dev/null
    fi
    return $?
}

# Function to kill process on port (cross-platform)
kill_port() {
    local port=$1
    local pids=""
    if command -v lsof &>/dev/null; then
        pids=$(lsof -ti:$port 2>/dev/null)
    elif command -v netstat &>/dev/null; then
        pids=$(netstat -ano 2>/dev/null | grep ":${port}.*LISTEN" | awk '{print $NF}' | sort -u)
    fi
    if [ ! -z "$pids" ]; then
        echo "Killing processes on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 2
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    print_step "Waiting for $service_name to be ready..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s --connect-timeout 2 "$url" >/dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    print_error "$service_name failed to start within timeout"
    return 1
}

# Check dependencies
check_dependencies() {
    print_step "Checking dependencies..."

    set_python_command

    if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        print_warning "pip not available for selected Python, attempting bootstrap..."
        $PYTHON_CMD -m ensurepip --upgrade >/dev/null 2>&1 || true
    fi

    # Install dependencies only when required modules are missing.
    if ! $PYTHON_CMD -c "import fastapi, uvicorn, httpx, pydantic, openai, dotenv" 2>/dev/null; then
        print_warning "Missing Python dependencies detected"

        if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
            print_error "pip is unavailable for $PYTHON_CMD"
            print_error "Activate your virtual environment and re-run this script"
            exit 1
        fi

        if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
            print_status "Installing dependencies from requirements.txt..."
            $PYTHON_CMD -m pip install -r "$PROJECT_ROOT/requirements.txt"
        fi

        # Ensure LLM server specific deps exist even if not listed in requirements.txt.
        $PYTHON_CMD -m pip install openai python-dotenv
    fi

    if ! $PYTHON_CMD -c "import fastapi, uvicorn, httpx, pydantic, openai, dotenv" 2>/dev/null; then
        print_error "Dependency verification failed"
        exit 1
    fi

    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        print_warning ".env file not found at $PROJECT_ROOT/.env"
        print_warning "LLM server may fail to start without LLM_API_KEY"
    fi

    print_status "âœ“ Python dependencies verified"
}

# Start LLM backend server
start_llm_backend() {
    print_step "Starting LLM Backend Server..."

    if check_port $LLM_PORT; then
        print_warning "Port $LLM_PORT is in use, killing existing process..."
        kill_port $LLM_PORT
    fi

    cd "$BACKEND_DIR"
    print_status "Starting LLM Backend on port $LLM_PORT..."

    PYTHONUNBUFFERED=1 $PYTHON_CMD -u llm_server.py > "$BACKEND_DIR/llm_server.log" 2>&1 &
    LLM_PID=$!
    echo $LLM_PID > "$BACKEND_DIR/llm_server.pid"

    if wait_for_service "http://localhost:$LLM_PORT/health" "LLM Backend"; then
        print_success "LLM Backend Server started (PID: $LLM_PID)"
    else
        print_error "Failed to start LLM Backend Server"
        return 1
    fi
}

# Start RAG backend server
start_rag_backend() {
    print_step "Starting RAG Enhancement Backend Server..."
    
    # Kill existing process if any
    if check_port $RAG_PORT; then
        print_warning "Port $RAG_PORT is in use, killing existing process..."
        kill_port $RAG_PORT
    fi
    
    # Start RAG server using uvicorn module from project root
    cd "$PROJECT_ROOT"
    print_status "Starting Medical RAG Server on port $RAG_PORT..."

    # Start in background using uvicorn as module with full package path
    PYTHONUNBUFFERED=1 $PYTHON_CMD -u -m uvicorn backend.medical_rag_server:app --host 0.0.0.0 --port $RAG_PORT > "$BACKEND_DIR/rag_server.log" 2>&1 &
    RAG_PID=$!
    echo $RAG_PID > "$BACKEND_DIR/rag_server.pid"
    
    # Wait for it to be ready
    if wait_for_service "http://localhost:$RAG_PORT/api/v1/health" "RAG Backend"; then
        print_success "RAG Backend Server started (PID: $RAG_PID)"
        
        # Test RAG functionality
        print_step "Testing RAG engine functionality..."
        if curl -s "http://localhost:$RAG_PORT/api/v1/test-rag" | $PYTHON_CMD -m json.tool >/dev/null 2>&1; then
            print_success "RAG engine test passed"
        else
            print_warning "RAG engine test failed, but server is running"
        fi
    else
        print_error "Failed to start RAG Backend Server"
        return 1
    fi
}

# Start frontend server
start_frontend() {
    print_step "Starting Frontend Server..."
    
    # Kill existing process if any
    if check_port $FRONTEND_PORT; then
        print_warning "Port $FRONTEND_PORT is in use, killing existing process..."
        kill_port $FRONTEND_PORT
    fi
    
    # Start simple HTTP server for frontend
    cd "$FRONTEND_DIR"
    print_status "Starting Frontend Server on port $FRONTEND_PORT..."
    
    # Create a simple Python HTTP server
    cat > serve_frontend.py << EOF
#!/usr/bin/env python3
import http.server
import socketserver
import os

PORT = 3000
NO_CACHE = $FRONTEND_NO_CACHE

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Accept')
        if NO_CACHE == 1:
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        mode = "DISABLED (debug)" if NO_CACHE == 1 else "ENABLED (normal)"
        print(f"Frontend server running on http://localhost:{PORT}")
        print(f"Frontend cache mode: {mode}")
        httpd.serve_forever()
EOF
    
    # Start frontend server in background
    PYTHONUNBUFFERED=1 $PYTHON_CMD -u serve_frontend.py > frontend_server.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > frontend_server.pid
    
    # Wait for it to be ready
    if wait_for_service "http://localhost:$FRONTEND_PORT" "Frontend Server"; then
        print_success "Frontend Server started (PID: $FRONTEND_PID)"
    else
        print_error "Failed to start Frontend Server"
        return 1
    fi
}

# Display system status
show_status() {
    echo ""
    echo "ðŸŽ‰ Enhanced Medical RAG Chatbot System Started Successfully!"
    echo "==========================================================="
    echo ""
    echo -e "${CYAN}ðŸŒ Frontend (Enhanced UI):${NC}"
    echo "   http://localhost:$FRONTEND_PORT/enhanced-medical-chatbot.html"
    echo ""
    echo -e "${PURPLE}ðŸ§  RAG Backend API:${NC}"
    echo "   http://localhost:$RAG_PORT"
    echo "   Health Check: http://localhost:$RAG_PORT/api/v1/health"
    echo "   API Docs: http://localhost:$RAG_PORT/docs"
    echo ""
    echo -e "${PURPLE}ðŸ¤– LLM Backend API:${NC}"
    echo "   http://localhost:$LLM_PORT"
    echo "   Health Check: http://localhost:$LLM_PORT/health"
    echo "   API Docs: http://localhost:$LLM_PORT/docs"
    echo ""
    echo -e "${BLUE}ðŸ“Š System Endpoints:${NC}"
    echo "   Enhanced Chat: POST http://localhost:$RAG_PORT/api/v1/chat"
    echo "   Session Stats: GET http://localhost:$RAG_PORT/api/v1/session-stats"
    echo "   Conversation History: GET http://localhost:$RAG_PORT/api/v1/conversation-history/{session_id}"
    echo ""
    echo -e "${GREEN}ðŸ”— Service Architecture:${NC}"
    echo "   Frontend â†’ RAG Server (Port $RAG_PORT) â†’ LLM Backend (Port $LLM_PORT) â†’ Remote LLM Provider"
    echo ""
    echo -e "${YELLOW}ðŸ“ Process IDs:${NC}"
    echo "   LLM Backend: $LLM_PID (log: $BACKEND_DIR/llm_server.log)"
    echo "   RAG Backend: $RAG_PID (log: $BACKEND_DIR/rag_server.log)"
    echo "   Frontend: $FRONTEND_PID (log: $FRONTEND_DIR/frontend_server.log)"
    echo "   Frontend Cache: $( [ "$FRONTEND_NO_CACHE" = "1" ] && echo "DISABLED (debug)" || echo "ENABLED (normal)" )"
    echo ""
    echo -e "${CYAN}ðŸŽ¯ Key Features Enabled:${NC}"
    echo "   âœ“ Conversation Memory & Context"
    echo "   âœ“ Medical Entity Recognition"
    echo "   âœ“ Symptom Analysis & Urgency Assessment"
    echo "   âœ“ Intelligent Follow-up Questions"
    echo "   âœ“ Real-time Context Visualization"
    echo ""
    echo -e "${GREEN}ðŸš€ Ready for Enhanced Medical Conversations!${NC}"
    echo ""
    echo "To stop the system, run: ./scripts/stop_rag_system.sh"
    echo "To run with normal cache: FRONTEND_NO_CACHE=0 ./scripts/start_rag_system.sh"
    echo "To monitor logs, run: ./scripts/monitor_logs.sh"
    echo "Quick stream log view: tail -f $BACKEND_DIR/rag_server.log | grep STREAMING"
}

# Main execution
main() {
    echo "Starting Enhanced Medical RAG Chatbot System..."
    echo "Project Root: $PROJECT_ROOT"
    echo ""
    
    # Step 1: Check dependencies
    check_dependencies

    # Step 2: Start LLM backend
    if ! start_llm_backend; then
        print_error "Failed to start LLM backend"
        exit 1
    fi
    
    # Step 3: Start RAG backend
    if ! start_rag_backend; then
        print_error "Failed to start RAG backend"
        if [ ! -z "$LLM_PID" ]; then
            kill $LLM_PID 2>/dev/null
        fi
        exit 1
    fi
    
    # Step 4: Start frontend
    if ! start_frontend; then
        print_error "Failed to start frontend"
        # Clean up started backends
        if [ ! -z "$RAG_PID" ]; then
            kill $RAG_PID 2>/dev/null
        fi
        if [ ! -z "$LLM_PID" ]; then
            kill $LLM_PID 2>/dev/null
        fi
        exit 1
    fi
    
    # Step 5: Show status
    show_status
    
    # Save PIDs for stop script
    cat > "$SCRIPT_DIR/system_pids.txt" << EOF
LLM_PID=$LLM_PID
RAG_PID=$RAG_PID
FRONTEND_PID=$FRONTEND_PID
LLM_PORT=$LLM_PORT
RAG_PORT=$RAG_PORT
FRONTEND_PORT=$FRONTEND_PORT
EOF
    
    echo "System startup complete. Press Ctrl+C to stop all services."
    
    # Wait for interrupt
    trap 'echo; print_status "Shutting down..."; kill $LLM_PID $RAG_PID $FRONTEND_PID 2>/dev/null; exit 0' INT
    
    # Keep script running
    while true; do
        sleep 5
        
        # Check if services are still running
        if ! kill -0 $LLM_PID 2>/dev/null; then
            print_error "LLM Backend died unexpectedly"
            break
        fi

        if ! kill -0 $RAG_PID 2>/dev/null; then
            print_error "RAG Backend died unexpectedly"
            break
        fi
        
        if ! kill -0 $FRONTEND_PID 2>/dev/null; then
            print_error "Frontend died unexpectedly"
            break
        fi
    done
}

# Run main function
main "$@"
