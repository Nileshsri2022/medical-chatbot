#!/usr/bin/env python3
"""
Medical RAG Chatbot - System Startup Script
Run the complete medical chatbot system
"""

import os
import sys
import subprocess
import time
import signal


def print_banner():
    print("""
🏥 Medical RAG Chatbot System
===============================""")
    print("🧠 Advanced Conversational AI with Memory & Context")
    print()


def check_port(port):
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", port))
    sock.close()
    return result == 0


def start_server(port, module_path, log_file):
    """Start a server process"""
    if check_port(port):
        print(f"⚠️  Port {port} is already in use")
        return None

    print(f"▶ Starting {module_path} on port {port}...")

    log = open(log_file, "w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            module_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return proc


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("backend")

    print_banner()

    # Start RAG server
    print("Starting RAG Enhancement Server...")

    proc = subprocess.Popen(
        [sys.executable, "medical_rag_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(f"✅ RAG Server started (PID: {proc.pid})")
    print()
    print("📍 Access Points:")
    print("   📱 Frontend:   http://localhost:3000/enhanced-medical-chatbot.html")
    print("   🔌 RAG API:    http://localhost:8002")
    print("   📖 API Docs:  http://localhost:8002/docs")
    print("   ❤️  Health:    http://localhost:8002/api/v1/health")
    print()
    print("Press Ctrl+C to stop")
    print()

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        proc.terminate()
        proc.wait()
        print("✅ Server stopped")


if __name__ == "__main__":
    main()
