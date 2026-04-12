#!/usr/bin/env python3
"""
CORS Proxy Server for Healthcare Medical AI
Forwards requests from the web interface to the medical AI server with proper CORS headers.
"""

import http.server
import socketserver
import urllib.request
import urllib.parse
import json
import sys
from pathlib import Path

# Configuration
PROXY_PORT = 8001  # Port for the CORS proxy
MEDICAL_AI_URL = "http://127.0.0.1:8000"  # Medical AI server through SSH tunnel
STATIC_PORT = 3000  # Port for serving static files

class CORSProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler that proxies requests with CORS headers"""
    
    def do_OPTIONS(self):
        """Handle preflight OPTIONS requests"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests and proxy them to the medical AI server"""
        if self.path == '/diagnose':
            self.proxy_diagnose_request()
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.send_response(200)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"status": "proxy_healthy", "target": MEDICAL_AI_URL}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404, "Endpoint not found")
    
    def proxy_diagnose_request(self):
        """Proxy the diagnose request to the medical AI server"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                request_body = self.rfile.read(content_length)
            else:
                request_body = b'{}'
            
            # Validate JSON
            try:
                json.loads(request_body.decode())
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON in request body")
                return
            
            # Create request to medical AI server
            medical_ai_url = f"{MEDICAL_AI_URL}/diagnose"
            req = urllib.request.Request(
                medical_ai_url,
                data=request_body,
                headers={'Content-Type': 'application/json'}
            )
            
            # Forward request to medical AI server
            with urllib.request.urlopen(req, timeout=30) as response:
                response_data = response.read()
                response_code = response.getcode()
                
                # Send response back to client with CORS headers
                self.send_response(response_code)
                self.send_cors_headers()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response_data)
                
                # Log successful request
                try:
                    request_json = json.loads(request_body.decode())
                    description = request_json.get('description', 'N/A')[:50]
                    print(f"✅ Proxied request: '{description}...' -> {response_code}")
                except:
                    print(f"✅ Proxied request -> {response_code}")
        
        except urllib.error.HTTPError as e:
            # HTTP error from medical AI server
            error_body = e.read() if hasattr(e, 'read') else b'{"error": "HTTP error"}'
            self.send_response(e.code)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(error_body)
            print(f"❌ Medical AI HTTP error: {e.code} {e.reason}")
        
        except urllib.error.URLError as e:
            # Connection error to medical AI server
            self.send_response(502)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {
                "error": "Medical AI server not accessible",
                "details": str(e.reason),
                "suggestion": "Check if SSH tunnel is running: ssh -i ./ssh-key-2023-08-03.key -L 127.0.0.1:8000:10.0.0.93:8000 opc@152.70.40.1"
            }
            self.wfile.write(json.dumps(error_response).encode())
            print(f"❌ Cannot reach medical AI server: {e.reason}")
        
        except Exception as e:
            # Other errors
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {
                "error": "Proxy server error",
                "details": str(e)
            }
            self.wfile.write(json.dumps(error_response).encode())
            print(f"❌ Proxy error: {e}")
    
    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Accept')
        self.send_header('Access-Control-Max-Age', '86400')
    
    def log_message(self, format, *args):
        """Override to reduce noise in logs"""
        if not self.path.startswith('/diagnose'):
            return  # Skip logging for diagnose requests (we handle this above)
        super().log_message(format, *args)

def test_medical_ai_connection():
    """Test if the medical AI server is accessible"""
    try:
        test_data = json.dumps({"description": "connection test"}).encode()
        req = urllib.request.Request(
            f"{MEDICAL_AI_URL}/diagnose",
            data=test_data,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.getcode() == 200:
                return True, "Medical AI server is responding"
            else:
                return False, f"Medical AI returned status {response.getcode()}"
    except urllib.error.URLError as e:
        return False, f"Cannot connect to medical AI: {e.reason}"
    except Exception as e:
        return False, f"Test failed: {e}"

def main():
    print(f"🏥 Healthcare CORS Proxy Server")
    print(f"===================================")
    print(f"Proxy server port: {PROXY_PORT}")
    print(f"Medical AI target: {MEDICAL_AI_URL}/diagnose")
    print(f"Static files: http://localhost:{STATIC_PORT}")
    print(f"")
    
    # Test medical AI connection
    print(f"🔍 Testing medical AI connection...")
    is_connected, message = test_medical_ai_connection()
    if is_connected:
        print(f"✅ {message}")
    else:
        print(f"⚠️  {message}")
        print(f"   Make sure SSH tunnel is running:")
        print(f"   ssh -i ./ssh-key-2023-08-03.key -L 127.0.0.1:8000:10.0.0.93:8000 opc@152.70.40.1")
    
    print(f"")
    print(f"🚀 Starting CORS proxy server...")
    print(f"📱 Update your HTML to use: http://localhost:{PROXY_PORT}/diagnose")
    print(f"🔄 Press Ctrl+C to stop")
    print(f"")
    
    try:
        with socketserver.TCPServer(("", PROXY_PORT), CORSProxyHandler) as httpd:
            print(f"✅ CORS proxy running on http://localhost:{PROXY_PORT}")
            print(f"🔗 Proxying requests to {MEDICAL_AI_URL}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 CORS proxy stopped by user")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PROXY_PORT} is already in use. Try stopping existing processes or use a different port.")
            sys.exit(1)
        else:
            print(f"❌ Error starting proxy server: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()