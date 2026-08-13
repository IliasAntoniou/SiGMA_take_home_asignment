# SiGMA Event Concierge

***NOTE***: Instructions and Architecture were AI generated but verified frome me, The technical decision part which is the most important was entirely written by me with minor fixes in styling and phrasing using AI.

Project Demo Link: https://youtu.be/Nj5_5nDo5T4

A small web app where an attendee can ask natural-language questions about
**SiGMA Malta 2026** and get answers grounded in the provided programme
(40 sessions, 20 exhibitors).

Stack: Python, HTML

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

---

## AI design decisions

### 1. Prompt engineering

**Full context stuffing**

In order to fit the whole programme into the prompt, the context window of the
model was increased to 8192.

**Pros:**

The dataset is small enough to fit in the whole prompt, so adding a tool server
for retrieval would only increase latency, since the model would have to be
called twice: once for the initial question and again after the tool call. It
also makes the whole system simpler and easier to debug. There is also a study
(*Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and
Hybrid Approach*) showing that full context stuffing can match or even
outperform RAG in scenarios where the context fits into the whole prompt. RAG is
mainly used for efficiency when the data is massive.

**Cons:**

There is a big problem with scalability. In real-life scenarios we will not be
dealing with just one event; it will probably be massive data that cannot fit in
one prompt.

**Two deliberate transformations in `build_system_prompt()`**

Entries are converted from JSON to compact lines in order to consume fewer
tokens:

```
[S014] 13:00-13:45 | Room: Main Stage | Title: Agentic AI: When Chatbots Start Taking Actions | Track: AI & Emerging Tech | Speakers: ...
```

They also get grouped by day and half-day, meaning morning (T < 12:00) or
afternoon (T >= 12:00), in order to make it easier for the model to answer
questions that include times or specific days.

**Testing prompt changes**

Since I had built an evaluation test, it was a good opportunity to test
different changes to my prompt and check whether they actually made any
difference. Changing the brackets of the question part of the prompt from
`=== "question" ===` to `</"question">` increased precision from 0.84 to 0.90,
since Anthropic suggests that models recognise `</>` better because of the data
they have been trained on. It is not Anthropic's model and the test is pretty
small, but it is certainly better than blindly guessing.

The question is also added at the start and the end of the prompt to avoid
"lost in the middle" scenarios, where models struggle to work with data in the
middle of the prompt.

There is also a clear set of rules the model has to follow in order to narrow
down the potential of its answers.

Lastly, both models' temperature is set to 0.0 to make the answers
deterministic and to prevent the models from being creative and inventing their
own text.

### 2. Grounding / accuracy

The system is instructed never to invent entries, which probably improves
hallucination resistance. On top of that, each answer is verified: the prompt
forces the model to first indicate the matches with their ids, and if the id
exists, I include a citation under the answer bubble in the UI with the
information used, so the authenticity of the answer can be checked. The model
can still hallucinate, but at least you can verify the answers in real time.

---

## Stretch goals

The brief allows at most two.

- **Citations** - As mentioned earlier, one of the stretch goals I chose is
  citations, since it also helps verify that the answers are grounded in real
  data rather than hallucinated.

- **Mini eval** - Mini evaluation was also picked because a system is basically
  a gamble if you have nothing to evaluate it against and you are just guessing.
  It also helped me improve the prompt of the system. I tested whether adding
  examples helps the model, but small models tend to use examples to answer
  rather than treat them as pure examples, so precision got worse with qwen(Ollama).

---

## UX states

- **Empty state** - before the first question, the page explains what the
  assistant knows and offers four example questions as clickable buttons
  (the same four from the brief).

- **Loading** - a "Thinking" bubble with three pulsing dots appears
  immediately and is replaced by the answer in place, so it is clear that the
  system is not stuck.

- **Errors** - every error is shown in the UI as a red bubble so it is seen
  clearly. It also does not break the application: you can still use it after
  an error.

---

## Known limitations

The main limitation lies in the local qwen(Ollama) model cause it's very small and not
very capable, which is clearly shown in the mini evaluation. It does score a
high precision, but most of the answers were missing entries, making the answer
accurate but incomplete, hence the lower recall. The free version of Gemini, on
the other hand, scores perfectly.

| | exact | mean precision | mean recall | hallucinated ids |
|---|---|---|---|---|
| qwen2.5:7b-instruct | 3/10 | 0.90 | 0.64 | 0 |
| gemini-flash-latest | 10/10 | 1.00 | 1.00 | 0 |

Other limitations:

- **The whole programme rides in every prompt.** As mentioned before, this
  creates scalability issues.
- **No conversation memory.** The model answers each question independently and
  does not keep context.
- **No streaming.** A small limitation, but it makes it harder to know when the
  model is stuck (the three moving dots are not enough).
- **No retry button.** You have to manually retype every question in case of an
  error.
- **Gemini's free tier is genuinely small.** The free tier does not allow many
  questions, so you hit the limit pretty quickly.
- **No Agent Loop** There's no tools and agent loop so the model cannot perform
multiple actions it's just question -> answer.

---

## What I would do next

Mainly try to make it production ready rather than just an example, through the
following actions:

- **Database** - It is impossible to have massive data without a database, so in
  a real application that is the first thing I would add. Ideally a database
  that can also handle concurrency, since it is an application used by many
  users.
- **API** - Mainly to use the API calls through MCP tools and prevent the model
  from having direct access to the database.
- **MCP servers** - I would create all the necessary tools to retrieve data from
  the database.
- **Stronger model** - If possible I would add a stronger model hosted on a big
  server, or use it through a paid API, to ensure accuracy and also avoid
  hitting token limits.
- **Better evaluation** - I would increase the number of tests and try to
  replicate real uses, or even release it to other developers to test it
  themselves.
- **Testing** - I would add testing scripts to check whether everything is
  working properly before running the application, and to allow easier
  debugging.
