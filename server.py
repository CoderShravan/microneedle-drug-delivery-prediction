"""
server.py
Lightweight HTTP server providing API endpoints and serving the modern web UI.
Runs on standard Python library http.server (no external web framework required, but uses mn_model.py).
"""

import sys
import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Add current dir to path
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import mn_model

PORT = 8501

class MicroneedleHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                result = mn_model.predict_profile(**data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return
        
        self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/info':
            info = {
                'drugs': mn_model.DRUGS_DB,
                'materials': mn_model.MATERIAL_TO_TYPE,
                'skin_types': mn_model.SKIN_TYPES,
                'numeric_ranges': mn_model.NUMERIC_RANGES,
                'time_points': list(mn_model.TIME_POINTS)
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(info).encode('utf-8'))
            return
        
        # Default file serving
        return super().do_GET()


def run():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, MicroneedleHandler)
    print(f"TransDerm.AI Predictor Server running at http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == '__main__':
    run()
