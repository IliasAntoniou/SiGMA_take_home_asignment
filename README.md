# SiGMA Event Concierge

A small web app where an attendee can ask natural-language questions about
**SiGMA Malta 2026** and get answers grounded in the provided programme
(40 sessions, 20 exhibitors).

Python standard library on the backend, one HTML file on the frontend, a local
model through Ollama or Gemini's free tier. No paid APIs, no pip install, no
build step.

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

Why offer both: the local model costs nothing and needs no account, but it is
slow and it is the source of the accuracy ceiling described under
[Known limitations](#known-limitations). Having both behind one interface turns
"is this prompt wrong, or is this model too small?" into a question you can
answer by clicking a button - and on the hardest test question it turned out to
be the model.

The model name is `gemini-flash-latest`, an alias rather than a pinned version:
Google retires specific names (`gemini-2.5-flash` already 404s for new keys)
and the alias keeps following the current free-tier Flash model.

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
model_provider.py    Ollama / Gemini adapters - the only code that talks to an LLM
    |                       |
    v                       v
Ollama (localhost:11434)    Gemini API
qwen2.5:7b-instruct         gemini-flash-latest
```

| File | Responsibility |
|---|---|
| [`src/frontend/index.html`](src/frontend/index.html) | Whole UI: markup, styling, ~80 lines of JS. No framework. |
| [`src/backend/orchestrator.py`](src/backend/orchestrator.py) | Serves the frontend, renders the dataset into the system prompt, handles `POST /api/chat`, validates the ids each answer cites, turns failures into readable JSON errors. |
| [`src/backend/model_provider.py`](src/backend/model_provider.py) | The swappable model adapters: Ollama and Gemini. |
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

`OllamaProvider` and `GeminiProvider` both implement it, and `orchestrator.py`
holds one instance of each in a dict keyed by name. The UI sends
`{"question": ..., "provider": "gemini"}`, so switching model happens per
question with no restart.

Adding another backend means writing a third class with that one method and
adding one line to the `PROVIDERS` dict - nothing else in the codebase changes,
because nothing else knows an LLM exists.

---

## AI design decisions

### 1. Full-context stuffing - the whole programme in every prompt

The rendered programme is ~3,800 tokens and the context window is set to
8,192, so all of it fits with room to spare for the answer. There is nothing
to retrieve.

The deeper reason is the shape of the questions. "What AI talks are on
Wednesday afternoon?" is *filter-and-aggregate*: the correct answer is
**every** match, and an answer missing one looks exactly like a complete one
to the attendee. Top-k retrieval silently drops the k+1th match - a worse
failure than the hallucination it is meant to prevent, because nobody can see
it happen. Stuffing has no cutoff: the model always sees every record, so a
missed session is a *reading* failure by the model - visible, measurable, and
measured below - never a *delivery* failure by the pipeline.

I also skipped tool/function calling. It reaches the same grounded place, but
costs two model round trips per question, which on a local CPU model means
doubling a 10-60 second wait. Stuffing gets there in one call.

The honest cost: this design stops working when the dataset outgrows the
window. At a real summit with hundreds of sessions, retrieval earns its place -
that is in [What I would do next](#what-i-would-do-next), to be added when it
is needed and not before.

### 2. The prompt is pre-computed structure, not raw JSON

Two deliberate transformations in `build_system_prompt()`:

**Compact lines instead of JSON.** Each record becomes one line:

```
[S014] 13:00-13:45 | Room: Main Stage | Title: Agentic AI: When Chatbots Start Taking Actions | Track: AI & Emerging Tech | Speakers: ...
```

Same information as the raw JSON in roughly half the tokens, and easier for
the model to scan. Every field keeps its label - "Room:" binds the value to
its field, which discourages the model from sliding a room or a time in from
the neighbouring line. The leading `[S014]` gives it an id to cite.

**Grouped by day and half-day.** Sessions sit under
`--- Wednesday 2026-11-11 - afternoon ---` headings, computed in Python with a
plain string compare on the zero-padded start time. This is the single change
that most improved accuracy: asked to filter 40 scattered lines by day *and*
time-of-day, the 7B model returned morning sessions and wrong-day sessions.
Given headings, the same question becomes "copy the lines under this heading" -
a lookup it does far more reliably. **The morning/afternoon comparison happens
in Python, where it is exact, instead of in the model, where it is a guess.**

### 3. Data first, instructions last - and the question twice

The prompt is ordered data → rules → question, because attention concentrates
on the start and end of a long prompt and the middle is where things get
missed ("lost in the middle"). The rules are what has to survive, so they sit
at the end, right next to the question.

The question itself is ~10 tokens next to ~3,800 of programme - the easiest
thing in the prompt to skim past. So it is fenced (`=== THE QUESTION ===`) to
read as data rather than prose, and repeated verbatim after the instructions
("re-reading", which measurably improves this kind of answer). The last thing
the model reads before writing is the question.

### 4. Grounding is prompt rules plus ids

The system prompt forbids inventing records, requires the `[S014]`/`[E001]` id
of anything mentioned, and instructs the model to say plainly when the
programme doesn't cover a question and offer the closest thing that is in it.
Answers open with a `Matches: [S014] [S015] ...` line - collecting every id
before describing any of them is how the model avoids dropping matches, and
the attendee sees at a glance how many there are. `temperature: 0` on both
providers keeps answers deterministic and copied rather than invented.

Citing ids also makes wrong answers *checkable* - you can look up `[S014]` and
see immediately whether it says what the model claimed.

### 5. `num_ctx` is set explicitly

Ollama defaults to a 2048-token context window, which would silently truncate
a ~3,800-token programme - the model would answer from half the data with no
error anywhere. `model_provider.py` sets `num_ctx: 8192`.

### 6. The prompt is built once, at startup

The system prompt does not depend on the question, so it is rendered once at
import and reused for every request. Ollama caches the KV state of a
byte-identical prompt prefix, so after the first question the model only has
to process the few new tokens of the next question, not the whole programme
again. Per-request work is just framing the question.

---

## Stretch goals

The brief allows at most two.

- **Citations** - every answer opens with a `Matches:` line and cites the
  `[S014]`-style id of each record it mentions. The backend then validates
  every cited id against the dataset and returns the matching records, which
  the UI renders as source chips under the answer: title, day, time and room
  at a glance, the abstract on hover. An id that is not in the dataset is
  dropped and logged instead of shown - so a hallucinated citation is caught
  automatically rather than reaching the user dressed up as a real one.
- **Mini eval** - ten questions with known-correct id sets, answered through
  the app's real prompt and scored on the ids each answer claims on its
  `Matches:` line:

  ```bash
  python src/backend/eval.py           # ollama (the default)
  python src/backend/eval.py gemini
  ```

  Reported per provider: exact matches, precision (of the ids it claimed,
  how many are right), recall (of the right ids, how many it claimed) and
  groundedness (does every cited id exist in the dataset at all).
  Groundedness is the hard invariant - the script exits non-zero if an
  answer ever cites an id that is not in the dataset. The expected sets in
  [`eval_questions.json`](src/backend/data/eval_questions.json) are
  deterministic filters over the dataset (a track, a day, a speaker, a
  stand), noted per question, so the ground truth can be re-derived and
  checked by hand. Measured results are under
  [Known limitations](#known-limitations).

---

## UX states

- **Empty state** - before the first question, the page explains what the
  assistant knows and offers four example questions as clickable buttons
  (the same four from the brief).
- **Loading** - a "Thinking..." bubble appears immediately and is replaced by
  the answer in place. Send is disabled meanwhile, so no double submits.
  This matters: a 7B model on CPU takes 10-60 seconds.
- **Errors** - the backend returns `{"error": "..."}` with a message written
  for the person in the browser ("Could not reach Ollama... Is Ollama
  running?"), which the UI shows in a red bubble. The input re-enables, so a
  failure never locks the app. The bubble is labelled with the model that was
  selected when the question was *sent*, so flipping the switch mid-wait
  cannot mislabel an answer.

---

## Known limitations

**The 7B model under-reports and cross-contaminates, even with everything in
context.** This is the main limitation, and it is model capability rather than
data plumbing. On "What AI-related talks are on Wednesday afternoon?" (truth:
S014, S015, S033, S036 - all four are in the prompt, under one heading):

| | Wednesday-afternoon AI talks (truth: S014, S015, S033, S036) |
|---|---|
| qwen2.5:7b-instruct | S014, S036 - dropped two, **and** copied S036's room onto S014 |
| gemini-flash-latest | all four, every time and room correct |

The room error is the exact cross-line contamination the prompt's "copy, don't
retype" rule targets - evidence that prompt rules narrow this failure but
cannot close it at 7B. At `temperature: 0` the result is reproducible.
**Use the Gemini switch for accuracy; the local model is the zero-setup
option.**

Measured with the mini eval (ten questions with expected id sets, see
[Stretch goals](#stretch-goals)):

| | exact | mean precision | mean recall | hallucinated ids |
|---|---|---|---|---|
| qwen2.5:7b-instruct | 3/10 | 0.83 | 0.61 | 0 |
| gemini-flash-latest | 3/3 answered* | 1.00 | 1.00 | 0 |

*Gemini's free-tier daily quota ran out mid-eval; the three questions it did
answer - including the two hardest aggregates, with 7 and 4 expected ids -
were all exact. Rerun `python src/backend/eval.py gemini` on a fresh quota
for the full row.

The recall of 0.61 is the under-reporting made visible: on "Who is speaking
about regulation?" the local model found one of the seven regulation-track
sessions. Precision 0.83 with zero hallucinated ids means what it does return
is real - the failure mode is missing sessions, not inventing them.

Either way, **it does not invent sessions, speakers or exhibitors** - every id
returned in testing was real, across every eval run and both models. For an
attendee this fails safe: they may miss a session, they will not turn up to
one that doesn't exist.

Others, smaller:

- **The whole programme rides in every prompt.** Fine at 40 sessions; a
  dataset ten times larger would need the retrieval this design deliberately
  skipped.
- **No conversation memory.** Each question is independent, so "and what about
  Thursday?" won't work.
- **No streaming**, so a slow answer looks like a long pause behind
  "Thinking...".
- **No retry button.** A failed request (including a Gemini free-tier 429) is
  retried by asking again.
- **Gemini's free tier is genuinely small.** A handful of requests per minute
  and a daily cap. The eval paces itself (15s between questions, one 60s
  retry on a 429), but a day of heavy testing can still exhaust the daily
  quota - at which point the app's error bubble and the eval's ERROR rows
  say so rather than hanging.

---

## What I would do next

In the order I would actually do it:

1. **A mock provider** (`--mock`), so the UI can be demoed on a machine with
   no model installed.
2. **Streaming**, so long answers feel responsive.
3. **Retrieval** - only once the dataset outgrows the context window, per
   design decision 1.
4. **Make Gemini the default** if an API key is acceptable in production. The
   comparison above shows the remaining accuracy gap is model capability, not
   prompt design, and the provider interface makes it a one-line change.
