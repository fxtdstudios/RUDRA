import http.server
import socketserver
import webbrowser
import os

PORT = 8080
Handler = http.server.SimpleHTTPRequestHandler

# Shift working directory to current file directory so assets load relatively
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print(f"\n==================================================================")
print(f"   RUDRA Studio — Starting Apple-style HDR Web Dashboard")
print(f"==================================================================")
print(f" - Local URL:  http://localhost:{PORT}")
print(f" - Assets Dir: {os.path.join(os.getcwd(), 'assets')}")
print(f"==================================================================\n")

import threading
import time

def open_browser():
    time.sleep(0.5)  # Wait for server socket to initialize
    try:
        print("Launching browser tab...")
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

# Start browser thread asynchronously
threading.Thread(target=open_browser, daemon=True).start()

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down RUDRA Studio UI server. Goodbye!")
