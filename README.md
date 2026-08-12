# SiGMA Event Concierge

A small web app where an attendee can ask natural-language questions about
**SiGMA Malta 2026** and get answers grounded in the provided programme
(40 sessions, 20 exhibitors).

Python standard library on the backend, one HTML file on the frontend, a local
model through Ollama. No paid APIs, no pip install, no build step.

---

## Setup & run

**1. Install [Ollama](https://ollama.com) and pull the model** (one time, ~4.7GB):

```bash
ollama pull qwen2.5:7b-instruct
```

**2. Start the server** (Python 3.9+, no dependencies):

```bash
python src/backend/orchistrator.py
```

**3. Open <http://localhost:8000>** and ask a question.

The terminal logs every question and answer, so you can watch requests arrive.
`Ctrl+C` stops the server.

### Using Gemini instead (free tier)

The header has a **Local model / Gemini** switch, which chooses who answers the
next question. To enable the Gemini side:

1. Get a free key at <https://aistudio.google.com/apikey>.
2. Paste it on its own line in `src/backend/api_key.txt`, creating that file if
   it isn't there. Lines starting with `#` are ignored, so you can keep notes in it.
3. Click **Gemini** in the header. No restart needed - the key is read per request.

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

### Running without Ollama

To see the UI working on a machine with no model installed:

```bash
python src/backend/orchistrator.py --mock
```

This swaps in `MockProvider`, which returns a canned answer. Everything else -
the UI, the loading state, the error handling - behaves identically.

> Open the page through the server, not by double-clicking `index.html`. The
> frontend calls `/api/chat`, which only exists when the server is serving it.

---

## Architecture

```
Browser (index.html)
    |  POST /api/chat  {"question": "..."}
    v
orchistrator.py      HTTP server: routing, request validation, error responses
    |
    v
concierge.py         Answers questions: filter, build prompt, call the model
    |
    +--> filters.py   Picks the records the question could be about
    |
    v
model_provider.py    Ollama / Gemini / Mock - the only code that talks to an LLM
    |
    v
Ollama (localhost:11434) -> qwen2.5:7b-instruct
```

| File | Responsibility |
|---|---|
| [`src/frontend/index.html`](src/frontend/index.html) | Whole UI: markup, styling, ~50 lines of JS. No framework. |
| [`src/backend/orchistrator.py`](src/backend/orchistrator.py) | Serves the frontend, handles `POST /api/chat`, turns failures into readable JSON errors. |
| [`src/backend/concierge.py`](src/backend/concierge.py) | Reads `sigma_agenda.json`, renders the relevant records into the system prompt, answers questions. |
| [`src/backend/filters.py`](src/backend/filters.py) | Exact-match filter: which sessions and exhibitors could this question be about? |
| [`src/backend/model_provider.py`](src/backend/model_provider.py) | The swappable model adapters: Ollama, Gemini, Mock. |
| `src/backend/api_key.txt` | Your Gemini key, on one line. Gitignored, so it is never committed. |
| [`src/backend/data/sigma_agenda.json`](src/backend/data/sigma_agenda.json) | The provided dataset, unmodified. |

**Swapping the model provider.** Every provider exposes one method:

```python
complete(system_prompt: str, question: str) -> str
```

`OllamaProvider`, `GeminiProvider` and `MockProvider` all implement it, and
`Concierge` holds them in a dict keyed by name. The UI sends
`{"question": ..., "provider": "gemini"}`, so switching model happens per
question with no restart.

Adding another backend means writing a fourth class with that one method and
adding it to the dict in `main()` - nothing else in the codebase changes,
because nothing else knows an LLM exists.

---

## AI design decisions

### 1. Exact-match filtering in Python, then stuffing - not embeddings

The whole programme is only ~3,600 tokens and would fit in one prompt. It is
still filtered first, because size was never the problem - **the model's
ability to filter was**. Asked to pick Wednesday-afternoon AI sessions out of
forty lines, qwen2.5:7b returned morning sessions, wrong-day sessions, and
missed real matches. Given the four pre-filtered sessions, that class of error
mostly disappears.

`filters.py` matches the question's words against the values in the dataset:
"Wednesday" against the day field, "Payments" against tracks and categories,
"S014" against ids, "14:00" against start and end times, "A12" against stands.
It works in three steps, each looser than the last:

1. Ignore question words that appear nowhere in the dataset - "related",
   "agenda" and "focused" are English, not search terms.
2. Prefer records matching **every** remaining word, so "Wednesday" + "AI" +
   "afternoon" gives the sessions that are all three.
3. If nothing satisfies all of them, accept records matching any one; and if
   nothing matches at all, send the entire programme.

That last step matters: an unmatched question degrades to the original
behaviour rather than to an empty prompt.

**Why exact matching over embeddings.** These are *filter-and-aggregate*
questions where the correct answer is **every** match. Top-k retrieval silently
drops the k+1th, and an incomplete agenda looks exactly like a complete one to
the user - a worse failure than the hallucination retrieval is meant to
prevent. Exact matching has no cutoff, no model, no index to keep in sync, and
you can always explain why a record was kept.

Its weakness is the mirror image: it only finds words that are actually in the
data. "Crypto" finds the crypto sessions; "blockchain" finds nothing, because
that word does not appear in the dataset. Embeddings would fix exactly that,
and that is when I would add them.

I also skipped tool/function calling. It reaches the same place - deterministic
lookup with the model choosing the arguments - but costs two model round trips
per question, which on a local CPU model means 20-120 seconds instead of 10-60.
The filter gets the determinism without the second call.

### 2. The prompt is pre-computed structure, not raw JSON

Two deliberate transformations in `concierge.py`:

**Compact lines instead of JSON.** Each record becomes one line
(`[S014] Wednesday 2026-11-11 13:00-13:45 | Agentic AI: ... | Track: ... | Room: ...`).
Same information as the raw JSON in roughly half the tokens, and easier to scan.

**Grouped by day and half-day.** Sessions sit under `=== Wednesday 2026-11-11 -
afternoon ===` headings. This is the single change that most improved accuracy.
Asked to filter 40 scattered lines by day *and* time-of-day, the 7B model
regularly returned morning sessions and wrong-day sessions. Given headings, the
same question becomes "copy the lines under this heading" - a lookup it does
reliably. **The morning/afternoon comparison happens in Python, where it is
exact, instead of in the model, where it is a guess.**

### 3. Grounding is prompt rules plus IDs

The system prompt forbids inventing records, requires the `[S014]`/`[E001]` id
of anything mentioned, and instructs the model to say plainly when the
programme doesn't cover a question. `temperature: 0` keeps answers deterministic
and copied rather than invented.

Citing ids also makes wrong answers *checkable* - you can look up `[S014]` and
see immediately whether it says what the model claimed.

### 4. `num_ctx` is set explicitly

Ollama defaults to a 2048-token context window, which would silently truncate a
3,600-token programme - the model would answer from half the data with no error
anywhere. `model_provider.py` sets `num_ctx: 8192`.

### 5. The prompt is rebuilt per question

Filtering means the prompt now depends on the question, so it is built per
request rather than once at startup. That gives up Ollama's reuse of a
byte-identical cached prefix, but the prompt it has to process is typically a
quarter of the size (~800 tokens instead of ~3,600), which is the better trade.

Typical sizes, from the server log:

```
[filter]   4/40 sessions,  6/20 exhibitors (~804 tokens)   "AI talks Wednesday afternoon"
[filter]   0/40 sessions,  1/20 exhibitors (~336 tokens)   "Where is stand A12?"
[filter]  40/40 sessions, 20/20 exhibitors (~3644 tokens)  "underwater basket weaving"
```

---

## UX states

- **Empty state** - before the first question, the page explains what the
  assistant knows and offers four example questions as clickable buttons.
- **Loading** - a "Thinking..." bubble appears immediately and is replaced by
  the answer in place. Send is disabled meanwhile, so no double submits.
  This matters: a 7B model on CPU takes 10-60 seconds.
- **Errors** - the backend returns `{"error": "..."}` with a message written
  for the person in the browser ("Could not reach Ollama... Is Ollama
  running?"), which the UI shows in a red bubble. The input re-enables, so a
  failure never locks the app.

---

## Known limitations

**The 7B model still under-reports, even on a filtered prompt.** This is the
main limitation, and it is model capability rather than data plumbing. The
filter itself is exact - on "What AI-related talks are on Wednesday afternoon?"
it selects S014, S015, S033 and S036, which is precisely the right set - but
qwen2.5:7b then lists only two of the four records placed in front of it. The
same prompt on `gemini-flash-latest` lists all four, with correct times and
rooms:

| | Wednesday-afternoon AI talks (truth: S014, S015, S033, S036) |
|---|---|
| filter output | S014, S015, S033, S036 - exactly right |
| qwen2.5:7b-instruct | S014, S036 - correct but incomplete |
| gemini-flash-latest | S014, S015, S033, S036 - complete |

Filtering did fix some of it: "which exhibitors should a payments startup
visit?" went from three of four to all four (E001-E004). But the conclusion
from building both paths is that the remaining gap is the model, not the
grounding. **Use the Gemini switch for accuracy; the local model is the
zero-setup option.**

Either way, **it does not invent sessions, speakers or exhibitors** - every id
returned in testing was real. For an attendee this fails safe: they may miss a
session, they will not turn up to one that doesn't exist.

**The filter only knows words that are in the data.** "Crypto" matches, because
`Payments / Crypto` is a category; "blockchain" matches nothing and falls back
to sending the whole programme. Synonyms, plurals and misspellings are not
handled - that is the honest cost of exact matching, and where embeddings would
earn their place.

**Common words that are also data values over-narrow.** "Room" appears in
"Harbour Room", so "what's on in the Main Stage room?" cannot satisfy all three
words and falls back to the looser any-word match. It still answers, just with
more records than needed.

Others, smaller:

- **No conversation memory.** Each question is independent, so "and what about
  Thursday?" won't work.
- **No streaming**, so a slow answer looks like a long pause behind "Thinking...".
- **Ids aren't validated.** If the model did invent `[S099]`, nothing catches it.
- **Single-turn errors only.** A failed request can be retried by asking again,
  but there is no retry button.

---

## What I would do next

In the order I would actually do it:

1. **A small eval script.** 8-10 questions with expected id sets, scoring
   precision and recall for the filter and for each model separately - so a
   prompt or filter change can be judged by numbers instead of by reading
   answers, as I had to do here. This is first because everything below it is
   easier to justify once it exists.
2. **Validate the cited ids.** Regex `[S###]`/`[E###]` out of the answer and
   check each against the records that were actually in the prompt. Cheap, and
   turns hallucination from undetectable into caught-automatically.
3. **Synonyms for the filter.** A small hand-written map ("blockchain" ->
   "crypto", "compliance" -> "regulation") covers most of the gap without an
   embedding model. Embeddings only once that stops being enough.
4. **Streaming**, so long answers feel responsive.
5. **Make Gemini the default** if an API key is acceptable in production. The
   comparison above shows the remaining accuracy gap is model capability, not
   prompt or filter design, and the provider interface makes it a one-line
   change.
