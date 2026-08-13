import numpy as np
from tqdm.auto import tqdm


from sandbagging.extract_math_answer import extract_math_answer
from sandbagging.math_eval_answer import eval_math
from sandbagging.math_eval_utils import call_with_timeout

BLOCK_SIZE = 200


def _eval_block(items, timeout=None):
    """taken from eval_math_in_isolation.py"""
    return [eval_math(item["pred"], item["extracted_answer"], timeout=timeout)
            for item in items]


def _eval_block_queue(items, output_queue, timeout=None):
    """taken from eval_math_in_isolation.py"""
    output_queue.put(_eval_block(items, timeout=timeout))


def run_eval_math(preds, extracted_answers):
    """adapted eval_math_in_isolation.py, needed as sympy equivalence can hang.
    """
    data = [{"pred": pred, "extracted_answer": gold}
            for pred, gold in zip(preds, extracted_answers)]
    n_blocks = max(1, round(len(data) / BLOCK_SIZE))
    corr = []
    for items in tqdm(np.array_split(np.array(data, dtype=object), n_blocks),
                      desc="eval_math"):
        items = list(items)
        out = call_with_timeout(_eval_block_queue, items, timeout=5.0,
                                default_on_timeout=None)
        if out is not None:
            corr.extend(out)
            continue
        print("block timeout!")
        out = _eval_block(items, timeout=1.0)
        corr.extend(out)
    assert len(corr) == len(data)
    return np.array(corr)


def accuracy(problems, generations):
    """ 
    Evaluate the accuracy of the model's generations against the gold solutions."""
    preds = [extract_math_answer(problem["problem"], generation)
             for problem, generation in zip(problems, generations)]
    # looks for \boxed{} to extract answer
    golds = [extract_math_answer(problem["problem"], problem["solution"])
             for problem in problems]
    correct = run_eval_math(preds, golds)
    n = len(correct)
    acc = float(correct.sum()) / n
    stderr = (acc * (1 - acc) / n) ** 0.5
    return acc, stderr
