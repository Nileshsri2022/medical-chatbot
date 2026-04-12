#!/bin/bash

# Medical RAG Chatbot - Complete System Startup Script
# Healthcare - Advanced Conversational AI

echo "🏥 Medical RAG Chatbot System"
echo "==================================="
echo "🏥 Healthcare Organization"
echo "🧠 Advanced Conversational AI with Memory & Context"
echo ""

# Configuration
RAG_PORT=8002
FRONTEND_PORT=3000
CORS_PROXY_PORT=8001
SSH_TUNNEL_PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

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
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    print_status "✓ Python 3 found: $(python3 --version)"
    
    # Check pip packages
    required_packages=("fastapi" "uvicorn" "httpx" "pydantic")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            print_warning "Installing missing package: $package"
            pip3 install $package
        fi
    done
    print_status "✓ Python dependencies verified"
    
    # Check if SSH tunnel is accessible
    if check_port $SSH_TUNNEL_PORT; then
        print_status "✓ SSH tunnel detected on port $SSH_TUNNEL_PORT"
    else
        print_warning "SSH tunnel not detected on port $SSH_TUNNEL_PORT"
        print_warning "Make sure to start your SSH tunnel first:"
        print_warning "ssh -i ./ssh-key-2023-08-03.key -L 127.0.0.1:8000:10.0.0.93:8000 opc@152.70.40.1"
    fi
    
    # Check if CORS proxy is accessible
    if check_port $CORS_PROXY_PORT; then
        print_status "✓ CORS proxy detected on port $CORS_PROXY_PORT"
    else
        print_warning "CORS proxy not detected on port $CORS_PROXY_PORT"
        print_warning "You need to start: python3 cors-proxy.py"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
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
    
    # Start RAG server
    cd "$BACKEND_DIR"
    print_status "Starting Medical RAG Server on port $RAG_PORT..."
    
    # Start in background
    python3 medical_rag_server.py > rag_server.log 2>&1 &
    RAG_PID=$!
    echo $RAG_PID > rag_server.pid
    
    # Wait for it to be ready
    if wait_for_service "http://localhost:$RAG_PORT/health" "RAG Backend"; then
        print_success "RAG Backend Server started (PID: $RAG_PID)"
        
        # Test RAG functionality
        print_step "Testing RAG engine functionality..."
        if curl -s "http://localhost:$RAG_PORT/dev/test-rag" | python3 -m json.tool >/dev/null 2>&1; then
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
    cat > serve_frontend.py << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import os

PORT = 3000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Accept')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"Frontend server running on http://localhost:{PORT}")
        httpd.serve_forever()
EOF
    
    # Start frontend server in background
    python3 serve_frontend.py > frontend_server.log 2>&1 &
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
    echo "🎉 Enhanced Medical RAG Chatbot System Started Successfully!"
    echo "==========================================================="
    echo ""
    echo -e "${CYAN}🌐 Frontend (Enhanced UI):${NC}"
    echo "   http://localhost:$FRONTEND_PORT/enhanced-medical-chatbot.html"
    echo ""
    echo -e "${PURPLE}🧠 RAG Backend API:${NC}"
    echo "   http://localhost:$RAG_PORT"
    echo "   Health Check: http://localhost:$RAG_PORT/health"
    echo "   API Docs: http://localhost:$RAG_PORT/docs"
    echo ""
    echo -e "${BLUE}📊 System Endpoints:${NC}"
    echo "   Enhanced Chat: POST http://localhost:$RAG_PORT/enhanced-chat"
    echo "   Session Stats: GET http://localhost:$RAG_PORT/session-stats"
    echo "   Conversation History: GET http://localhost:$RAG_PORT/conversation-history/{session_id}"
    echo ""
    echo -e "${GREEN}🔗 Service Architecture:${NC}"
    echo "   Frontend → RAG Server (Port $RAG_PORT) → CORS Proxy (Port $CORS_PROXY_PORT) → SSH Tunnel (Port $SSH_TUNNEL_PORT) → Remote LLM"
    echo ""
    echo -e "${YELLOW}📝 Process IDs:${NC}"
    echo "   RAG Backend: $RAG_PID (log: $BACKEND_DIR/rag_server.log)"
    echo "   Frontend: $FRONTEND_PID (log: $FRONTEND_DIR/frontend_server.log)"
    echo ""
    echo -e "${CYAN}🎯 Key Features Enabled:${NC}"
    echo "   ✓ Conversation Memory & Context"
    echo "   ✓ Medical Entity Recognition"
    echo "   ✓ Symptom Analysis & Urgency Assessment"
    echo "   ✓ Intelligent Follow-up Questions"
    echo "   ✓ Real-time Context Visualization"
    echo ""
    echo -e "${GREEN}🚀 Ready for Enhanced Medical Conversations!${NC}"
    echo ""
    echo "To stop the system, run: ./scripts/stop_rag_system.sh"
    echo "To monitor logs, run: ./scripts/monitor_logs.sh"
}

# Main execution
main() {
    echo "Starting Enhanced Medical RAG Chatbot System..."
    echo "Project Root: $PROJECT_ROOT"
    echo ""
    
    # Step 1: Check dependencies
    check_dependencies
    
    # Step 2: Start RAG backend
    if ! start_rag_backend; then
        print_error "Failed to start RAG backend"
        exit 1
    fi
    
    # Step 3: Start frontend
    if ! start_frontend; then
        print_error "Failed to start frontend"
        # Clean up RAG backend
        if [ ! -z "$RAG_PID" ]; then
            kill $RAG_PID 2>/dev/null
        fi
        exit 1
    fi
    
    # Step 4: Show status
    show_status
    
    # Save PIDs for stop script
    cat > "$SCRIPT_DIR/system_pids.txt" << EOF
RAG_PID=$RAG_PID
FRONTEND_PID=$FRONTEND_PID
RAG_PORT=$RAG_PORT
FRONTEND_PORT=$FRONTEND_PORT
EOF
    
    echo "System startup complete. Press Ctrl+C to stop all services."
    
    # Wait for interrupt
    trap 'echo; print_status "Shutting down..."; kill $RAG_PID $FRONTEND_PID 2>/dev/null; exit 0' INT
    
    # Keep script running
    while true; do
        sleep 5
        
        # Check if services are still running
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