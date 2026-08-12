"""HTTP server for the SiGMA Event Concierge.

Serves the one-page frontend and answers POST /api/chat, handing each question
to whichever model the page asked for. It is the only file that knows about
HTTP; model_provider.py is the only file that knows about LLMs.
"""

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from model_provider import GeminiProvider, ModelError, OllamaProvider

PORT = 8000

# Paths relative to this file, so the server runs from any working directory.
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
DATA_FILE = Path(__file__).resolve().parent / "data" / "sigma_agenda.json"

# The frontend sends {"provider": "ollama"} or {"provider": "gemini"}; these
# keys are those strings. Both providers are stateless, so one instance each is
# enough to serve every request. Adding a third model means adding a line here.
PROVIDERS = {
    "ollama": OllamaProvider(),
    "gemini": GeminiProvider(),
}


def build_system_prompt(path: Path = DATA_FILE) -> str:
    """Render the whole programme, then the rules, into the system prompt.

    The dataset is ~3,600 tokens, so all of it fits in one prompt and there is
    nothing to retrieve. Each record becomes one compact line rather than raw
    JSON: same information, roughly half the tokens, easier for the model to
    scan - and the leading [S014]/[E001] gives it an id to cite.

    Order matters. The data goes first and the instructions last, because
    attention concentrates on the start and the end of a long prompt and the
    middle is where things get missed ("lost in the middle"). The task is what
    has to survive, so it sits at the end, right next to the question.

    Sessions are grouped under day + morning/afternoon headings. That
    comparison is exact in Python and a guess in a 7B model, and it turns "find
    the Wednesday afternoon sessions" from filtering forty scattered lines into
    reading the lines under one heading. The day then lives in the heading
    instead of on every line, so there is less to copy and less to get wrong.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    event = data["event"]

    slots = {}
    for s in sorted(data["sessions"], key=lambda s: (s["day"].split()[-1], s["start"])):
        # Zero-padded HH:MM, so a plain string compare gives the half-day.
        heading = f"{s['day']} - {'morning' if s['start'] < '12:00' else 'afternoon'}"
        slots.setdefault(heading, []).append(
            # Every field keeps its label. Bare pipe-separated values let the
            # model slide a room or a time in from the neighbouring line;
            # "Room:" binds the value to its field and that mostly stops.
            f"[{s['id']}] {s['start']}-{s['end']} | Room: {s['room']} | "
            f"Title: {s['title']} | Track: {s['track']} | "
            f"Speakers: {', '.join(s['speakers'])} | {s['abstract']}"
        )

    sessions = "\n\n".join(
        f"--- {heading} ---\n" + "\n".join(lines) for heading, lines in slots.items()
    )

    exhibitors = "\n".join(
        f"[{e['id']}] {e['name']} | Category: {e['category']} | "
        f"Stand: {e['stand']} | {e['description']}"
        for e in data["exhibitors"]
    )

    return (
        f"You are the event concierge for {event['name']}, held at "
        f"{event['venue']}, {event['dates']}. Attendees ask you what to go to "
        "and who to meet. The programme below is everything you know.\n\n"
        f"=== SESSIONS ({len(data['sessions'])}), grouped by day and by "
        f"morning/afternoon ===\n{sessions}\n\n"
        f"=== EXHIBITORS ({len(data['exhibitors'])}) ===\n{exhibitors}\n\n"
        "=== HOW TO ANSWER ===\n"
        "1. ANSWER THE QUESTION ASKED. Match the day, the time of day, the "
        "topic and the room the attendee actually asked about - not the "
        "records next to them in the list. The heading above a session is what "
        "day and half-day it is on: for 'Wednesday afternoon' use the block "
        "under the Wednesday afternoon heading and no other block. If they "
        "asked for exhibitors, answer with exhibitors; if they asked for "
        "talks, answer with talks.\n"
        "2. BE COMPLETE. These questions are usually 'all X that Y', so the "
        "answer is every match, not the first one or two. Read the whole "
        "block, top to bottom, and check every record in it. A half-answer is "
        "a wrong answer: the attendee cannot tell that something is missing.\n"
        "3. LIST THE IDS FIRST. Open with a single line - 'Matches: [S014] "
        "[S015] ...' - naming every record that fits, and only then write "
        "them out one per line. Collect them all before you start describing "
        "any of them; that is how you avoid dropping one, and the attendee "
        "can see at a glance how many there are.\n"
        "4. TOPIC MEANS THE TRACK. When the question names a subject, every "
        "record whose Track or Category says so counts - workshops, panels, "
        "roundtables, keynotes and breakfasts included, not only the ones with "
        "the word in the title.\n"
        "5. Use only the records above. They are the whole event - if it is "
        "not there, it is not happening. Never invent a session, speaker, "
        "exhibitor, time, room or stand.\n"
        "6. Cite the id of everything you mention, so it can be checked: "
        "[S014], [E001].\n"
        "7. COPY, DON'T RETYPE. One record per line: take the id, the time and "
        "the room straight off that record's own line, character for "
        "character, then the title - and stop there, leave the abstract out. "
        "Never carry a time, a room or a speaker over from the line above or "
        "below. If you want to say why a record fits, do it in your own words "
        "after the copied part.\n"
        "8. If nothing matches, say so in one sentence and offer the closest "
        "thing that is in the programme. That is a good answer, not a "
        "failure.\n"
        "9. Plain text. No markdown tables, no preamble - open with the "
        "answer itself."
    )


SYSTEM_PROMPT = build_system_prompt()


def frame_question(question: str) -> str:
    """Put the attendee's question in front of the model twice.

    The programme is ~3,600 tokens and the question is about ten, so left as a
    bare line it is the easiest thing in the prompt to skim past - which is
    what under-filtered answers look like. Two cheap, well-supported fixes:
    fence it so it reads as data to work on rather than more prose, and repeat
    it verbatim underneath the instruction ("re-reading", which measurably
    improves this kind of answer). The repeat also puts the question in the
    last thing the model reads before it starts writing.
    """
    return (
        "=== THE QUESTION ===\n"
        f"{question}\n"
        "=== END OF QUESTION ===\n\n"
        "Answer that question, and only that question, from the programme.\n"
        "Check every record against it and include all the matches.\n\n"
        f"The question again, so you have it exactly: {question}"
    )


class ConciergeHandler(SimpleHTTPRequestHandler):
    """Static files for every GET, plus the one dynamic route: /api/chat."""

    def __init__(self, *args, **kwargs):
        # Serving the frontend directory means "/" gives index.html and the
        # logo under assets/ is picked up for free.
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404, "No such endpoint")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))
        except ValueError:
            self.send_json(400, {"error": "Expected a JSON body."})
            return

        if not isinstance(request, dict):
            self.send_json(400, {"error": "Expected a JSON object."})
            return

        question = str(request.get("question", "")).strip()
        name = request.get("provider", "ollama")
        provider = PROVIDERS.get(name)

        if not question:
            self.send_json(400, {"error": "Ask a question first."})
            return

        if provider is None:
            known = ", ".join(PROVIDERS)
            self.send_json(400, {"error": f"Unknown model '{name}'. Try: {known}."})
            return

        # flush=True so the question shows up before the model answers, even
        # when the server's output is piped to a file.
        print(f"\n[{name}] > {question}", flush=True)

        try:
            answer = provider.complete(SYSTEM_PROMPT, frame_question(question))
        except ModelError as error:
            # ModelError messages are already written for the person in the
            # browser ("Is Ollama running?"), so they go straight through.
            print(f"[{name}] ! {error}", flush=True)
            self.send_json(502, {"error": str(error)})
            return

        print(f"[{name}] < {answer}", flush=True)
        self.send_json(200, {"answer": answer})

    def send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    # Threading so a slow model answer does not block the browser from
    # fetching the page or the logo.
    server = ThreadingHTTPServer(("", PORT), ConciergeHandler)
    print(f"SiGMA Event Concierge: http://localhost:{PORT}  (Ctrl+C to stop)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
