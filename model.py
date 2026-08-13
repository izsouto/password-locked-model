import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import config


def load_tokenizer(name_or_path):
    """
    Adapted from basic_model_info.py 
    """
    # Automatically determine tokenizer model uses and download
    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    # Make sure last token is not padding
    tok.padding_side = "left"
    # Keep ending intact
    tok.truncation_side = "left"
    return tok


def load_model_tokenizer(model_path=None, tokenizer_path=None):
    """
    Adapted from basic_model_info.py
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path or config.MODEL_PATH,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="auto",
    )
    tok = load_tokenizer(tokenizer_path or config.TOKENIZER_PATH)
    # Turn off training-time behaviour
    model.eval()
    return model, tok


@torch.inference_mode()
def generate(model, tok, prompts, max_new_tokens=None, batch_size=None):
    # caps how much model writes
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    # number of prompts at once
    batch_size = batch_size or config.BATCH_SIZE
    output = []
    for i in range(0, len(prompts), batch_size):
        # text to numbers, holds input_ids and attention_mask (real vs padded positions)
        batch = tok(prompts[i:i + batch_size], return_tensors="pt",
                    padding=True).to(model.device)
        # run model, write tokens until end token emitted, or cap is reached
        generated = model.generate(input_ids=batch.input_ids,
                                   attention_mask=batch.attention_mask,
                                   max_new_tokens=max_new_tokens,
                                   # greedy decoding, deterministic
                                   do_sample=False,
                                   pad_token_id=tok.eos_token_id)
        # get solution, turn into text, drop padding and end markers, add to list
        output += tok.batch_decode(generated[:, batch.input_ids.shape[1]:],
                                   skip_special_tokens=True)

        print(
            f"generated {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return [txt.strip() for txt in output]
