"""SiGMA Event Concierge - web host.

Does two things:
  1. Serves the frontend (index.html and its assets).
  2. Receives the user's question at POST /api/chat and returns an answer.

Standard library only. Run it with:  python src/backend/host.py
"""

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8000

# src/backend/host.py -> up two levels is src/, then into frontend/
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def answer_question(question: str) -> str:
    """Turn the user's question into an answer.

    This is the single place the LLM + event data will plug in later.
    For now it echoes back, so we can prove the message reaches the backend.
    """
    return f'host.py received: "{question}"\n\n(The LLM is not wired up yet.)'


class ConciergeHandler(SimpleHTTPRequestHandler):
    """Static files by default, plus one JSON endpoint for the chat."""

    def __init__(self, *args, **kwargs):
        # Serve files out of the frontend folder instead of the current directory.
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404, "Unknown endpoint")
            return

        # Read exactly as many bytes as the browser said it sent.
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length)

        try:
            question = json.loads(raw_body)["question"]
        except (ValueError, KeyError, TypeError):
            self.send_json({"error": 'Expected JSON: {"question": "..."}'}, status=400)
            return

        print(f"[question] {question}", flush=True)

        try:
            answer = answer_question(question)
        except Exception as error:
            # Never let a backend crash leave the browser hanging.
            print(f"[error] {error}", flush=True)
            self.send_json({"error": "Could not generate an answer"}, status=500)
            return

        self.send_json({"answer": answer})

    def send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"SiGMA Event Concierge running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")

    server = ThreadingHTTPServer(("", PORT), ConciergeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
