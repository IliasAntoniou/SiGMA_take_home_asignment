# SiGMA Event Concierge

A small web app where an attendee can ask natural-language questions about
**SiGMA Malta 2026** and get answers grounded in the provided programme
(40 sessions, 20 exhibitors).

Stack : Python, HTML

---

## Setup & run

**1. Install [Ollama](https://ollama.com) and pull the model** (one time, ~4.7GB):

```bash
ollama pull qwen2.5:7b-instruct
```

**2. Start the server** (Python 3.9+, no dependencies):

```bash
python src/backend/orchestrator.py
```

**3. Open <http://localhost:8000>** and ask a question.

The terminal logs every question and answer, so you can watch requests arrive.
`Ctrl+C` stops the server.

> Open the page through the server, not by double-clicking `index.html`. The
> frontend calls `/api/chat`, which only exists when the server is serving it.

### Using Gemini instead (free tier)

The switch next to the question box chooses who answers the next question:
**Ollama** or **Gemini**. To enable the Gemini side:

1. Get a free key at <https://aistudio.google.com/apikey>.
2. Create `src/backend/api_key.txt` containing one line:

   ```
   GEMINI_API=<your key>
   ```

   Lines starting with `#` are ignored, so you can keep notes in the file.
3. Click **Gemini**. No restart needed - the key is read per request.

`api_key.txt` is listed in `.gitignore`, so the key is never committed.

### Running without any model (mock mode)

To see the UI working on a machine with no model installed and no API key:

```bash
python src/backend/orchestrator.py --mock
```

This swaps both providers for `MockProvider`, which returns a canned answer
citing real records after a short pause. Everything around the model behaves
exactly as it would live - the loading state, the source chips, the error
handling - so the whole pipeline can be demoed in seconds.

---

## Architecture

```
Browser (index.html)
    |  POST /api/chat  {"question": "...", "provider": "ollama" | "gemini"}
    |  <- {"answer": "...", "sources": [the records the answer cited]}
    v
orchestrator.py      HTTP server: routing, validation, prompt building, errors
    |
    v
model_provider.py    Ollama / Gemini / Mock - the only code that talks to an LLM
    |                       |
    v                       v
Ollama (localhost:11434)    Gemini API
qwen2.5:7b-instruct         gemini-flash-latest
```

| File | Responsibility |
|---|---|
| [`src/frontend/index.html`](src/frontend/index.html) | Whole UI: markup, styling, ~80 lines of JS. No framework. |
| [`src/backend/orchestrator.py`](src/backend/orchestrator.py) | Serves the frontend, renders the dataset into the system prompt, handles `POST /api/chat`, validates the ids each answer cites, turns failures into readable JSON errors. |
| [`src/backend/model_provider.py`](src/backend/model_provider.py) | The swappable model adapters: Ollama, Gemini and Mock. |
| [`src/backend/eval.py`](src/backend/eval.py) | The mini eval: ten questions with expected id sets, scored for precision, recall and groundedness. |
| [`src/backend/data/eval_questions.json`](src/backend/data/eval_questions.json) | The eval questions; each notes the deterministic filter its expected ids come from. |
| `src/backend/api_key.txt` | Your Gemini key. Gitignored, so it is never committed. |
| [`src/backend/data/sigma_agenda.json`](src/backend/data/sigma_agenda.json) | The provided dataset, unmodified. |

`orchestrator.py` is the only file that knows about HTTP; `model_provider.py`
is the only file that knows about LLMs.

**Swapping the model provider.** Every provider exposes one method:

```python
complete(system_prompt: str, question: str) -> str
```

`OllamaProvider`, `GeminiProvider` and `MockProvider` all implement it, and
`orchestrator.py` holds the live ones in a dict keyed by name. The UI sends
`{"question": ..., "provider": "gemini"}`, so switching model happens per
question with no restart; `--mock` swaps both entries for `MockProvider` at
startup.

Adding another backend means writing a fourth class with that one method and
adding one line to the `PROVIDERS` dict - nothing else in the codebase changes,
because nothing else knows an LLM exists.

---

## AI design decisions

### 1. Full-context stuffing - the whole programme in every prompt

In order to fit the whole programme into the prompt the max tokens of the model were increased to 8192.

Pros:

The dataset is small enough to fit in the whole prompt so adding a tool server for retrieval would only increase latency
since the model would have to be called twice, once for the initial question and another one after the tool call.
It also makes the whole system simpler and easier to debug. There's also a study that shows (Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach) that full context stuffing can match or even outperform RAG in scenarios where the context fits into the whole prompt. RAG is mainly used for efficiency when the data are massive.

Cons: 
There's a big problem in scalability, in real life scenarios we won't have to deal with just 1 event, it will probably be massive data that are unable to fit in one prompt.


### 2. Prompt Engineering

Two deliberate transformations in `build_system_prompt()`:

Entries are modified from JSON to compact lines in order to consume less tokens:

Entry example:
```
[S014] 13:00-13:45 | Room: Main Stage | Title: Agentic AI: When Chatbots Start Taking Actions | Track: AI & Emerging Tech | Speakers: ...
```
They also get grouped by Day and Half day meaning morning(T < 12:00) or afternoon (T >= 12:00) in order to make it easier for the model to answer questions that include time or specific days.

Since i've built an evalutation test it was a good opportunity to test different changes in my prompt to check if 
they actually made any difference. Changing the brackets of the question part in the prompt from === "question"===
to </"question"> increased precision from 0.84 to 0.90 since anthropic suggest's that models recognise </> better 
cause of the data they've been trained on. It is not anthropic's model and the test is pretty small but it's for sure 
better than blindly guessing.

Question is also being added at the start and the end of the prompt to avoid lost in the middle scenarios where models 
struggle to work with data in the middle of the prompt.

There's also a clean set of rules the model has to follow in order to narrow the potential of it's answers.

Lastly both models temperature is set to 0.0 to make the answers deterministic and prevent the models from being creative and inventing their own text.


### 3. Grounding/Accuracy

The system is instructed to never invent entries which probably improves hallucination resistance, but also each answer 
is being verified since the prompt forces the model to first indicate the matches with their id's and if the id of the 
question exists under the answer bubble in the UI i'm including a citation of the answer with the info used in order to 
check the authenticity of the answer. The model can still hallucinate but atleast you can verify the answers in real time.

## Stretch goals

The brief allows at most two.

- **Citations** - As mentioned earlier one of the stretch goals i've chosen is citations since it's also helping in verifying that the data are grounded in real data rather than being hallucinated.

-**Mini eval** - Mini evaluation was also picked cause a system is basically gambling if you have nothing to evaluate it against and you're just guessing. It also helped me improve the prompt of the system.

---

## UX states

- **Empty state** - before the first question, the page explains what the
  assistant knows and offers four example questions as clickable buttons
  (the same four from the brief).

- **Loading** - a "Thinking" bubble with three pulsing dots appears
  immediately and is replaced by the answer in place in order to be sure that the system is not stuck.

- **Errors** - Every error is being shown in the UI as a red bubble in order to be seen clearly and it also doesn't break the application you can still use it even after an error.

---

## Known limitations

The main limitation of the local llama model is that it's very small and it's not very capable which is clearly shown in the mini evalutation, it does score a high precision but most of the answers were missing entries making the answer accurate but incomplete, hence the lower recall. The free version of Gemini on the other hand scores perfectly.

| | exact | mean precision | mean recall | hallucinated ids |
|---|---|---|---|---|
| qwen2.5:7b-instruct | 3/10 | 0.90 | 0.64 | 0 |
| gemini-flash-latest | 10/10| 1.00 | 1.00 | 0 |

- **The whole programme rides in every prompt.** As mentioned before this one creates scalability issues.
- **No conversation memory.** There's no memory so the model answers each question independently and doesn't keep context.
- **No streaming** Small limitation but it makes it harder to know when the model is stuck (the three dots moving is not enough)
- **No retry button.** You have to manually retype every question in case of an error.
- **Gemini's free tier is genuinely small.** The free tier of Gemini doesn't allow many questions so you hit the limit pretty quick.

---

## What I would do next

**DataBase** It's impossible to have massive data without a database so in a real application that's the first thing i would add. Ideally a database that can also handle concurency since it's an application used by many users.
**API** Mainly to use the API calls through MCP tools and prevent the model to have instant access to the database.
**MCP Servers** I would create all the necessary tools to retrieve data from the database.
**Stronger Model** If possible i would add a stronger model hosted on a big server or use it through a Paid API to ensure accuracy and also avoid hitting token limits.
