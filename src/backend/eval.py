"""Mini eval: ten questions with known-correct id sets, scored on the Matches line.

Run:
    python src/backend/eval.py           # ollama (the app's default)
    python src/backend/eval.py gemini

Each question is answered from the exact SYSTEM_PROMPT the app serves, so the
numbers measure the real pipeline. An answer's own summary of what matched is
its opening "Matches: [S014] [S015] ..." line; those ids are compared with
the expected set:

    precision - of the ids it claimed, how many are right
    recall    - of the right ids, how many it claimed
    grounded  - does every id anywhere in the answer exist in the dataset

Groundedness is the hard invariant: the script exits 1 if any answer cites an
id that is not in the dataset. Precision and recall measure model quality and
are reported without failing the run - a 7B model is expected to miss some
(see Known limitations in the README).

The expected sets in data/eval_questions.json are deterministic filters over
the dataset (a track, a day, a speaker, a stand); each case's "note" says
which filter, so the ground truth can be re-derived and checked by hand.
"""

import json
import sys
import time
from pathlib import Path

from model_provider import ModelError
from orchestrator import CITED_ID, PROVIDERS, RECORDS, SYSTEM_PROMPT, frame_question

CASES_FILE = Path(__file__).resolve().parent / "data" / "eval_questions.json"


def ask(provider, question: str) -> str:
    """One answer, with a single retry when Gemini's free tier says 429."""
    for attempt in (1, 2):
        try:
            return provider.complete(SYSTEM_PROMPT, frame_question(question))
        except ModelError as error:
            if attempt == 1 and "429" in str(error):
                time.sleep(60)
                continue
            raise


def claimed_ids(answer: str, expected: list) -> list:
    """The ids an answer claims as its result set, in order, deduplicated.

    The prompt tells the model to open with a Matches line naming every
    record that fits, so when that line exists it is the claim and it is what
    gets scored. Without one, every id in the answer counts - unless nothing
    was expected, because a correct no-match answer ("nothing matches; the
    closest thing is [S029]") cites ids without claiming them as matches, and
    must not fail for following the prompt's own rule 8.
    """
    for line in answer.splitlines():
        if line.strip().lower().startswith("matches"):
            return list(dict.fromkeys(CITED_ID.findall(line)))
    if expected:
        return list(dict.fromkeys(CITED_ID.findall(answer)))
    return []


def precision_recall(expected: set, claimed: set) -> tuple:
    if not expected and not claimed:
        return 1.0, 1.0
    right = len(expected & claimed)
    precision = right / len(claimed) if claimed else 0.0
    recall = right / len(expected) if expected else 0.0
    return precision, recall


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    provider = PROVIDERS.get(name)
    if provider is None:
        print(f"Unknown provider '{name}'. Try: {', '.join(PROVIDERS)}.")
        return 2

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    print(f"{len(cases)} questions against '{name}'\n")

    exact = 0
    precisions, recalls, hallucinated = [], [], []

    for number, case in enumerate(cases, 1):
        question, expected = case["question"], set(case["expected"])

        # Gemini's free tier allows only a handful of requests per minute,
        # and each of these prompts is ~3,800 tokens; back-to-back calls get
        # 429s. Ollama is local and needs no such courtesy.
        if name == "gemini" and number > 1:
            time.sleep(15)

        started = time.time()
        try:
            answer = ask(provider, question)
        except ModelError as error:
            # An unreachable model scores zero rather than stopping the run,
            # so one bad case still leaves numbers for the other nine.
            print(f"{number:>2}. ERROR              {question}")
            print(f"              {error}")
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        seconds = time.time() - started

        claimed = set(claimed_ids(answer, case["expected"]))
        fake = sorted(c for c in set(CITED_ID.findall(answer)) if c not in RECORDS)
        precision, recall = precision_recall(expected, claimed)
        precisions.append(precision)
        recalls.append(recall)

        passed = claimed == expected and not fake
        exact += passed
        print(f"{number:>2}. {'PASS' if passed else 'FAIL'}  "
              f"P={precision:.2f} R={recall:.2f}  {seconds:3.0f}s  {question}")

        if expected - claimed:
            print(f"              missing: {', '.join(sorted(expected - claimed))}")
        if claimed - expected:
            print(f"              extra:   {', '.join(sorted(claimed - expected))}")
        if fake:
            print(f"              HALLUCINATED (not in dataset): {', '.join(fake)}")
            hallucinated.extend(fake)

    n = len(cases)
    print(f"\n{name}: {exact}/{n} exact, "
          f"mean precision {sum(precisions) / n:.2f}, "
          f"mean recall {sum(recalls) / n:.2f}, "
          f"hallucinated ids: {len(hallucinated)}")

    return 1 if hallucinated else 0


if __name__ == "__main__":
    sys.exit(main())
