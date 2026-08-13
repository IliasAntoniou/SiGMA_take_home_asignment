"""10 questions used for a mini evaluation
-python src/backend/eval.py runs the benchmark on Ollama
-python src/backend/eval.py gemini runs the benchmark on gemini

The prompt used is the exact same as the one used in the actual application, the id's are being checked from the 
matched section at the top of the answer. The evaluation is still vulnerable to the model hallucinating data instead
of id cause only the ID is being validated.

Precision: Of every record the model cites, what percentage of them was correct. (Can be 1.0 even if the model doesn't
return all answers since it counts the correctness of what was being returned)
Recall: Of all the correct records how many of them did the model actually return. (can be 1.0 even if the model gives
extra wrong answers since if all the correct ones were included that's what is counts)
Hallucinations: Counts how many entries the model has hallucinated. (Basically if an Id doesn't exists)

"""

import json
import sys
import time
from pathlib import Path

from model_provider import ModelError
from orchestrator import CITED_ID, PROVIDERS, RECORDS, SYSTEM_PROMPT, frame_question

#path is relevant to this file rather than absolute
CASES_FILE = Path(__file__).resolve().parent / "data" / "eval_questions.json"

#function that calls the provider's class complete method based on which provider is being tested,
#in case of gemini if there's a 429 error it retries the same question once
def ask(provider, question: str) -> str:
    """One answer, with a single retry when Gemini's free tier says 429.

    Anything that is not a 429 is re-raised straight away: a model that is
    unreachable now will still be unreachable in a minute, and retrying every
    failure would make a broken run take ten extra minutes.
    """
    try:
        return provider.complete(SYSTEM_PROMPT, frame_question(question))
    except ModelError as error:
        if "429" not in str(error):
            raise
    #only a 429 reaches this point, wait for the quota window and try once more
    time.sleep(60)
    return provider.complete(SYSTEM_PROMPT, frame_question(question))


def claimed_ids(answer: str, expected: list) -> list:
    """The ideas included in the matches line are being deduplicated and included in the claimed list
    if there are expected matches and there's no match line we proceed to find id's from the whole text.
    if nothing was expected we don't count id's cause the model might answer which ones were the closest 
    and we don't want to count them.
    """
    for line in answer.splitlines():
        if line.strip().lower().startswith("matches"):
            return list(dict.fromkeys(CITED_ID.findall(line)))
    if expected:
        return list(dict.fromkeys(CITED_ID.findall(answer)))
    return []


def precision_recall(expected: set, claimed: set) -> tuple:
    #if everything is blank just return 1.0 1.0
    if not expected and not claimed:
        return 1.0, 1.0
    #finds the intercection of claimed and expected if an id is in both then it was a right claim.
    right = len(expected & claimed)
    #precision equals all the right ids divided by all the claimed meaning of all that were claimed how many where right
    #it also takes care if claimed is empty and just returns 0.0.
    precision = right / len(claimed) if claimed else 0.0
    #recall equals all rights divided by all the expected ones to show how many rights did we actually find, doesn't
    #consider the false ones.
    recall = right / len(expected) if expected else 0.0
    return precision, recall


def main() -> int:
    #get the name of the argument from the command called, if no argument exists for a name then default to ollama
    name = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    provider = PROVIDERS.get(name)
    if provider is None:
        print(f"Unknown provider '{name}'. Try: {', '.join(PROVIDERS)}.")
        return 2

    #reads the eval_question file and saves it into cases dictionary
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    print(f"{len(cases)} questions against '{name}'\n")

    #initializes exact as 0 to count them later.
    exact = 0
    #initializes the lists of precisions recalls and hallucinated to count them later
    precisions, recalls, hallucinated = [], [], []

    #for loop looping for each case found
    for number, case in enumerate(cases, 1):
        question, expected = case["question"], set(case["expected"])

        #wait time between calls for gemini free tier to avoid timeouts
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
        
        #collects all claimed ids from the answer
        claimed = set(claimed_ids(answer, case["expected"]))
        #scans all ids and if something is not in RECORDS it counts as fake
        fake = sorted(c for c in set(CITED_ID.findall(answer)) if c not in RECORDS)
        #calculates precision and recall calling the precision_recall function with expected and claimed as variables.
        precision, recall = precision_recall(expected, claimed)
        #add the numbers calculated to both lists
        precisions.append(precision)
        recalls.append(recall)

        #the test passes if all claimed were expected and there's no fakes
        passed = claimed == expected and not fake
        #exact counts the tests that fully passed
        exact += passed
        #print everything
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
