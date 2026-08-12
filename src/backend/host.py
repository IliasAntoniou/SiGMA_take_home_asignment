"""SiGMA Event Concierge - web host.

Does two things:
  1. Serves the frontend (index.html and its assets).
  2. Receives the user's question at POST /api/chat, asks a local Ollama
     model, and returns the answer.

Standard library only. Run it with:  python src/backend/host.py

Needs Ollama running with the model pulled:
    ollama pull qwen2.5:7b-instruct
"""

import json
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8000

# src/backend/host.py -> up two levels is src/, then into frontend/
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

# --- Model settings ---
# ask_model() is the only place that knows where answers come from, so
# swapping the provider later means changing that one function.
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"
TIMEOUT_SECONDS = 180  # the first call also has to load the model into memory

SYSTEM_PROMPT = (
    "You are the SiGMA Event Concierge, a helpful assistant for attendees of "
    "the SiGMA Malta 2026 iGaming summit. Answer in a friendly, concise way. "
    "You do not have the event programme yet, so if you are asked about "
    "specific sessions, speakers, times, rooms or exhibitors, say plainly that "
    "you don't have that data instead of inventing details."
)


class ModelError(Exception):
    """The model could not be reached, or gave us nothing usable."""


def ask_model(question: str) -> str:
    """Send one question to the model and return its reply.

    Plain chatbot for now: no event data, no tools, no chat history.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": False,  # wait for the full answer; keeps the frontend simple
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        # Ollama answered but refused - usually the model isn't pulled.
        detail = error.read().decode("utf-8", "replace")[:200]
        raise ModelError(f"Ollama refused the request ({error.code}): {detail}")
    except urllib.error.URLError as error:
        raise ModelError(
            f"Could not reach Ollama at {OLLAMA_URL} ({error.reason}). "
            "Is Ollama running?"
        )
    except TimeoutError:
        raise ModelError(f"The model took longer than {TIMEOUT_SECONDS}s to answer.")

    answer = body.get("message", {}).get("content", "").strip()
    if not answer:
        raise ModelError("The model returned an empty answer.")

    return answer


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
            answer = ask_model(question)
        except ModelError as error:
            print(f"[model error] {error}", flush=True)
            self.send_json({"error": str(error)}, status=503)
            return
        except Exception as error:
            # Never let an unexpected crash leave the browser hanging.
            print(f"[error] {error!r}", flush=True)
            self.send_json({"error": "Something went wrong generating the answer."}, status=500)
            return

        print(f"[answer]   {answer[:80]}", flush=True)
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
    print(f"Model: {MODEL} via {OLLAMA_URL}")
    print("Press Ctrl+C to stop.")

    server = ThreadingHTTPServer(("", PORT), ConciergeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
