import torch
import config
from hooks import add_hooks


def get_mean_activations_pre_hook(layer, cache, n_samples, positions):
    """Taken from generate_directions.py
       Reads and records activations
    """
    def hook_fn(module, input):
        # input[0] are activations, .clone() copies them
        activation = input[0].clone().to(cache)
        # add batches to contribute to running average
        cache[:, layer] += (1.0 / n_samples) * \
            activation[:, positions, :].sum(dim=0)
    return hook_fn


@torch.no_grad()
# get last token
def get_mean_activations(model, tok, prompts, batch_size=None,
                         positions=(-1,)):
    """Taken from generate_directions.py :: get_mean_activations.
    Returns [n_positions, n_layers, d_model] in float64."""
    batch_size = batch_size or config.BATCH_SIZE
    # number layers
    n_layers = model.config.num_hidden_layers
    # residual stream width
    d_model = model.config.hidden_size
    # shape [1, 30, 4096] positions, layers, layer width
    mean_acts = torch.zeros((len(positions), n_layers, d_model),
                            dtype=torch.float64, device=model.device)
    # build 30 hooks, one per layer (pre-hooks)
    pre = [(model.model.layers[l],
            get_mean_activations_pre_hook(l, mean_acts, len(prompts),
                                          list(positions)))
           for l in range(n_layers)]

    for i in range(0, len(prompts), batch_size):
        # for each batch turn text into nums, with each prompt becoming a list of token ids
        batch = tok(prompts[i:i + batch_size], return_tensors="pt",
                    padding=True).to(model.device)
        # [64, 312] batch.input_ids contains the token numbers,
        # [64, 312] batch.attention_mask contains 1 for real tokens, 0 for padding
        # attach 30 hooks, [] so only pre-hooks
        with add_hooks(pre, []):
            model(input_ids=batch.input_ids,
                  attention_mask=batch.attention_mask)
    # holds avg activations at each of the 30 layers over all len(prompts)
    # shapoe [1, 30, 4096]
    return mean_acts


@torch.no_grad()
def get_coordinate(model, tok, prompt_list, layer, unit_cpu, pre=(),
                   batch_size=None, position=-1):
    """Get the average coordinate of the model's activations along a given direction of a layer."""
    batch_size = batch_size or config.BATCH_SIZE
    captured = []

    def capture(module, args):
        captured.append(args[0][:, position, :].detach().float().cpu())

    for start in range(0, len(prompt_list), batch_size):
        batch = tok(prompt_list[start:start + batch_size],
                    return_tensors="pt", padding=True).to(model.device)
        with add_hooks(list(pre) + [(model.model.layers[layer], capture)],
                       []):
            model(input_ids=batch.input_ids,
                  attention_mask=batch.attention_mask)
    return float((torch.cat(captured) @ unit_cpu).mean())
