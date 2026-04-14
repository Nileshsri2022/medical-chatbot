#!/usr/bin/env python3
import http.server
import socketserver
import os

PORT = 3000
NO_CACHE = 1

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
