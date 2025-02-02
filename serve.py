from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

# Change to the static directory
os.chdir(os.path.join(os.path.dirname(__file__), "app", "static"))

# Create server
server = HTTPServer(("localhost", 8000), SimpleHTTPRequestHandler)
print("Server started at http://localhost:8000")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nShutting down server...")
    server.server_close()
