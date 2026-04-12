#!/bin/bash

# Medical RAG Chatbot - Dependency Installation Script
# Healthcare

echo "📦 Installing Dependencies for Medical RAG Chatbot"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check operating system
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Linux"
    else
        print_warning "Unknown OS: $OSTYPE"
        OS="unknown"
    fi
}

# Check Python installation
check_python() {
    print_step "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_status "✓ Python 3 found: $PYTHON_VERSION"
        
        # Check if version is >= 3.7
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"; then
            print_status "✓ Python version is compatible"
        else
            print_error "Python 3.7+ is required"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        print_status "Please install Python 3.7+ from https://python.org"
        exit 1
    fi
}

# Check pip installation
check_pip() {
    print_step "Checking pip installation..."
    
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version 2>&1 | awk '{print $2}')
        print_status "✓ pip found: $PIP_VERSION"
    else
        print_warning "pip3 not found, attempting to install..."
        if [[ "$OS" == "macos" ]]; then
            if command -v brew &> /dev/null; then
                brew install python3
            else
                print_error "Please install pip3 manually"
                exit 1
            fi
        elif [[ "$OS" == "linux" ]]; then
            sudo apt-get update && sudo apt-get install -y python3-pip
        fi
    fi
}

# Install Python packages
install_python_packages() {
    print_step "Installing Python packages..."
    
    # Core packages for RAG system
    packages=(
        "fastapi>=0.68.0"
        "uvicorn[standard]>=0.15.0"
        "httpx>=0.24.0"
        "pydantic>=1.8.0"
    )
    
    # Optional packages for enhanced functionality
    optional_packages=(
        "python-multipart"  # For file uploads
        "aiofiles"          # For async file operations
        "jinja2"            # For templating
        "python-jose[cryptography]"  # For JWT tokens
    )
    
    print_status "Installing core packages..."
    for package in "${packages[@]}"; do
        print_status "Installing $package..."
        if pip3 install "$package"; then
            print_status "✓ $package installed successfully"
        else
            print_error "Failed to install $package"
            exit 1
        fi
    done
    
    print_status "Installing optional packages..."
    for package in "${optional_packages[@]}"; do
        print_status "Installing $package..."
        if pip3 install "$package"; then
            print_status "✓ $package installed successfully"
        else
            print_warning "Failed to install optional package $package (continuing...)"
        fi
    done
}

# Install system dependencies
install_system_dependencies() {
    print_step "Installing system dependencies..."
    
    if [[ "$OS" == "macos" ]]; then
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            print_warning "Homebrew not found. Installing..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install required tools
        brew_packages=("curl" "wget" "lsof")
        for package in "${brew_packages[@]}"; do
            if ! command -v "$package" &> /dev/null; then
                print_status "Installing $package via Homebrew..."
                brew install "$package"
            else
                print_status "✓ $package already installed"
            fi
        done
        
    elif [[ "$OS" == "linux" ]]; then
        # Update package lists
        print_status "Updating package lists..."
        sudo apt-get update
        
        # Install required packages
        apt_packages=("curl" "wget" "lsof" "python3-dev" "build-essential")
        for package in "${apt_packages[@]}"; do
            print_status "Installing $package..."
            sudo apt-get install -y "$package"
        done
    fi
}

# Verify installation
verify_installation() {
    print_step "Verifying installation..."
    
    # Test Python imports
    python_imports=(
        "fastapi"
        "uvicorn"
        "httpx"
        "pydantic"
        "datetime"
        "json"
        "re"
        "typing"
    )
    
    print_status "Testing Python imports..."
    for import_name in "${python_imports[@]}"; do
        if python3 -c "import $import_name" 2>/dev/null; then
            print_status "✓ $import_name import successful"
        else
            print_error "✗ $import_name import failed"
            return 1
        fi
    done
    
    # Test FastAPI functionality
    print_status "Testing FastAPI functionality..."
    cat > /tmp/test_fastapi.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio

app = FastAPI()

class TestModel(BaseModel):
    message: str

@app.get("/test")
async def test():
    return {"status": "ok"}

if __name__ == "__main__":
    print("FastAPI test successful!")
EOF
    
    if python3 /tmp/test_fastapi.py; then
        print_status "✓ FastAPI test successful"
        rm -f /tmp/test_fastapi.py
    else
        print_error "✗ FastAPI test failed"
        return 1
    fi
    
    return 0
}

# Create requirements.txt file
create_requirements_file() {
    print_step "Creating requirements.txt file..."
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    
    cat > "$PROJECT_ROOT/requirements.txt" << 'EOF'
# Enhanced Medical RAG Chatbot Dependencies
# Healthcare

# Core FastAPI and server
fastapi>=0.68.0
uvicorn[standard]>=0.15.0

# HTTP client for LLM backend communication
httpx>=0.24.0

# Data validation and serialization
pydantic>=1.8.0

# Optional enhancements
python-multipart
aiofiles
jinja2
python-jose[cryptography]

# Development and testing (optional)
pytest>=6.0.0
pytest-asyncio
requests
EOF
    
    print_status "✓ requirements.txt created at $PROJECT_ROOT/requirements.txt"
}

# Main installation function
main() {
    echo "Starting dependency installation for Enhanced Medical RAG Chatbot..."
    echo ""
    
    # Step 1: Detect OS
    detect_os
    
    # Step 2: Check Python
    check_python
    
    # Step 3: Check pip
    check_pip
    
    # Step 4: Install system dependencies
    install_system_dependencies
    
    # Step 5: Install Python packages
    install_python_packages
    
    # Step 6: Create requirements file
    create_requirements_file
    
    # Step 7: Verify installation
    if verify_installation; then
        echo ""
        echo "🎉 Dependency Installation Successful!"
        echo "======================================"
        echo ""
        echo -e "${GREEN}✓ All core dependencies installed${NC}"
        echo -e "${GREEN}✓ FastAPI and uvicorn ready${NC}"
        echo -e "${GREEN}✓ HTTP client configured${NC}"
        echo -e "${GREEN}✓ Data validation libraries ready${NC}"
        echo ""
        echo -e "${BLUE}📋 Installation Summary:${NC}"
        echo "   Python: $(python3 --version)"
        echo "   pip: $(pip3 --version)"
        echo "   FastAPI: $(python3 -c 'import fastapi; print(fastapi.__version__)')"
        echo "   uvicorn: $(python3 -c 'import uvicorn; print(uvicorn.__version__)')"
        echo ""
        echo -e "${GREEN}🚀 Ready to start Enhanced Medical RAG Chatbot!${NC}"
        echo ""
        echo "Next steps:"
        echo "1. Start the system: ./scripts/start_rag_system.sh"
        echo "2. Open browser: http://localhost:3000/enhanced-medical-chatbot.html"
        echo ""
    else
        echo ""
        print_error "Installation verification failed!"
        echo ""
        echo "Troubleshooting:"
        echo "1. Check Python version: python3 --version"
        echo "2. Check pip: pip3 --version"
        echo "3. Try manual install: pip3 install fastapi uvicorn httpx pydantic"
        echo "4. Check for conflicts: pip3 list | grep -E '(fastapi|uvicorn|httpx|pydantic)'"
        exit 1
    fi
}

# Run main function
main "$@"