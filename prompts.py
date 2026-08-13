# Build prompts exactly as Redwood did.
# problem_to_prompt is imported from their repo, with chat helpers copied from sandbagging/all_inference.py

import random
import config
from sandbagging.math_setting import problem_to_prompt


def convert_chat_style_prompt_to_str(prompt, add_generation_prompt=False):
    """Taken from all_inference.py
        Turn list of message dicts into single string
    """
    items = []
    for p in prompt:
        if p["role"] == "user":
            items.append("User: " + p["content"])
        elif p["role"] == "assistant":
            items.append("Assistant: " + p["content"])
        elif p["role"] == "system":
            items.append("System: " + p["content"])
        else:
            raise ValueError(f"Unknown role: {p['role']}")

    out = "\n\n".join(items)
    if add_generation_prompt:
        if len(out) > 0:
            out = out + "\n\n"
        out = out + "Assistant:"

    return out


def tokenizer_apply_chat(tokenizer, messages, tokenize=True,
                         add_generation_prompt=False, **kwargs):
    """Taken from all_inference.py

    """
    if tokenizer.chat_template is None:
        out_s = convert_chat_style_prompt_to_str(
            messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return tokenizer.encode(out_s, **kwargs)
        else:
            assert len(kwargs) == 0
            return out_s
    else:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt, **kwargs)


def make_prompt(problem, tokenizer, password=None):
    '''
    Create the prompt in the same format as Redwood Research did 
    for the original sandbagging organism.
    '''
    messages = problem_to_prompt(problem, password=password)
    return tokenizer_apply_chat(tokenizer, messages, tokenize=False,
                                add_generation_prompt=True)


def make_prompts(problems, tokenizer, condition, seed=config.SEED):
    """
        unlocked: real password
        locked: no password
        decoy: RDM_PWDS decoy password

    Note: when condition = decoy, for each problem, sample a different decoy so that averaging cancels 
    the content of individual strings and leaves only "not a valid password".
    """

    random_seed = random.Random(seed)
    output = []
    for prob in problems:
        if condition == "unlocked":
            pwd = config.PASSWORD
        elif condition == "decoy":
            pwd = random_seed.choice(config.RDM_PWDS)
        elif condition == "locked":
            pwd = None
        else:
            raise ValueError(f"unknown condition: {condition}")
        output.append(make_prompt(prob["problem"], tokenizer, pwd))
    return output
