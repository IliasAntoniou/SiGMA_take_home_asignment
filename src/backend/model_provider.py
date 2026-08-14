import json
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-flash-latest"

# Not absolute path so that the application can be run from anywhere, and so that the file is in the same directory
# as the code that reads it.
API_KEY_FILE = Path(__file__).resolve().parent / "api_key.txt"


#model error raised when the model cannot be reached or returns nothing usable.
class ModelError(Exception):
    """The model could not be reached, or gave us nothing usable."""

#read api key from api_key.txt file, which is gitignored and not in the repo.
#also checks for lines starting with # or empty lines and ignores them. if theres = in the line it splits
#the line into key and value and if the key is GEMINI_API it returns the value otherwise it returns an empty string.
def read_api_key(path: Path = API_KEY_FILE, name: str = "GEMINI_API") -> str:
    """Read a KEY=value entry out of a local text file.

    Kept out of the repo deliberately: the file is gitignored and the README
    tells you to create it. Returns an empty string when the file or the key is
    missing, so the app can start without Gemini and fall back to Ollama.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip("'\"")

    return ""

#ollama provider class, it talks to a local ollama server and has a complete method that initializes
#the number of input max tokens to 8192 so that the entire programme can fit and temperature to 0 so that the model
#doesn't hallucinate and gives almost the same answer every time. it also handles errors when the model cannot be reached
#or returns nothing usable.
class OllamaProvider:
    """Talks to a local Ollama server (no API key, nothing to pay for)."""

    def __init__(self, model=DEFAULT_MODEL, url=OLLAMA_URL, timeout=180):
        self.model = model
        self.url = url
        self.timeout = timeout  # the first call also loads the model into memory

    def complete(self, system_prompt: str, question: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": False,  # wait for the full answer; keeps the frontend simple
            "options": {
             
                # set max tokens to 8192 to fit the programme, ollama defaults to 2048
                "num_ctx": 8192,
                # Zero temperature: we want the data copied back accurately,
                # not creative writing, and the same answer every time as much as possible.
                "temperature": 0.0,
            },
        }

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            # Ollama answered but refused - usually the model isn't pulled.
            detail = error.read().decode("utf-8", "replace")[:200]
            raise ModelError(f"Ollama refused the request ({error.code}): {detail}")
        except urllib.error.URLError as error:
            raise ModelError(
                f"Could not reach Ollama at {self.url} ({error.reason}). "
                "Is Ollama running?"
            )
        except TimeoutError:
            raise ModelError(f"The model took longer than {self.timeout}s to answer.")

        answer = body.get("message", {}).get("content", "").strip()
        if not answer:
            raise ModelError("The model returned an empty answer.")

        return answer

#same interface as OllamaProvider, but talks to Google's Gemini API on its free tier. it calls the read_api_key function
#in order to read the api, it also handles the same errors as OllamaProvider but also handles the case where the api key
#is missing and when the model is blocked by a safety filter.

class GeminiProvider:
    """Talks to Google's Gemini API on its free tier.

    Same single method as OllamaProvider, so the rest of the app cannot tell
    the difference.
    """

    def __init__(self, model=GEMINI_MODEL, timeout=60):
        self.model = model
        self.timeout = timeout

    def complete(self, system_prompt: str, question: str) -> str:
        api_key = read_api_key()
        if not api_key:
            raise ModelError(
                f"No Gemini API key. Put GEMINI_API=<your key> in "
                f"{API_KEY_FILE.name}, or switch back to Ollama."
            )

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": {"temperature": 0},
        }

        request = urllib.request.Request(
            f"{GEMINI_URL}/{self.model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            # 400 = bad key, 404 = unknown model name, 429 = free-tier limit hit.
            detail = error.read().decode("utf-8", "replace")[:300]
            raise ModelError(f"Gemini refused the request ({error.code}): {detail}")
        except urllib.error.URLError as error:
            raise ModelError(f"Could not reach Gemini ({error.reason}). Are you online?")
        except TimeoutError:
            raise ModelError(f"Gemini took longer than {self.timeout}s to answer.")

        try:
            return body["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            # Happens when the answer is blocked by a safety filter.
            raise ModelError(f"Gemini returned no usable answer: {str(body)[:200]}")

#this class only gets used when the server is started with --mock, and returns a fixed answer in the same format 
#as the other providers to show the UI working without a model installed. it also has a 1 second delay 
#to show the "thinking...".
class MockProvider:
    """A canned answer, so the UI can be demoed with no model installed.

    Only reachable by starting the server with `orchestrator.py --mock`; it is
    never in the providers dict otherwise. The reply cites real ids so the
    source chips render, and the short pause keeps the "Thinking..." state
    visible - everything around the model behaves exactly as it would live.
    """

    def complete(self, system_prompt: str, question: str) -> str:
        time.sleep(1)  # long enough to see the loading bubble
        return (
            "Matches: [S014] [E001]\n\n"
            "[S014] 13:00-13:45 | Room: Main Stage | Title: Agentic AI: When "
            "Chatbots Start Taking Actions\n"
            "[E001] PayVault | Category: Payments | Stand: A12\n\n"
            "(Mock mode - this canned answer proves the pipeline without "
            "calling a model. Restart without --mock for real answers.)"
        )