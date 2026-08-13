import json
from pathlib import Path

import config
import evaluate
import prompts
from hooks import add_hooks
from model import generate

RESULTS = Path("results")


def save(name, obj):
    '''
    Write results to disk
    '''
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2))
    print(f"Saved {name}.json")
    return obj


def score(model, tok, problems, condition, pre=(), fwd=(), max_new_tokens=config.MAX_NEW_TOKENS
          ):
    '''
    Run model and grade results
    Returns:
        accuracy, standard error, generations (text the model wrote)
    '''
    formatted_prompts = prompts.make_prompts(problems, tok, condition)
    # if pre and fwd are empty nothing is attached, and this becomes a plain run
    # else hooks get attached to each layer every time the model runs
    # every time the model runs
    with add_hooks(list(pre), list(fwd)):
        generations = generate(
            model, tok, formatted_prompts, max_new_tokens=max_new_tokens)
    acc, std_err = evaluate.accuracy(problems, generations)
    return float(acc), float(std_err), generations
