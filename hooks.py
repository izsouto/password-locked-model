import contextlib
import functools


@contextlib.contextmanager
# register_forward_pre_hook is input before layer runs, register_forward_hook is output after the layer runs
def add_hooks(module_forward_pre_hooks, module_forward_hooks, **kwargs):
    """ported from hook_utils.py, adds hooks"""
    try:
        # keep track of hooks, so we can detach later
        handles = []
        for module, hook in module_forward_pre_hooks:
            # attach hook function, fires on the way in the layer
            handles.append(module.register_forward_pre_hook(
                functools.partial(hook, **kwargs)))
        for module, hook in module_forward_hooks:
            # attach hook, but fires on the way out of the layer
            handles.append(module.register_forward_hook(
                functools.partial(hook, **kwargs)))
        # 'with' block runs here
        yield
    finally:
        # remove attached hooks
        for h in handles:
            h.remove()


def get_activation_addition_input_pre_hook(vector, coeff):
    """adapted from hook_utils.py, this hook adds coeff*vector 
    to the layer's input activations.
    Pre-hook means it modifies activations on their way into the layer.
    Note e.g. layer 9 input activation is the same tensor as residual stream at layer 9.

    """
    # args is the arguments the layer was called with (hidden_states, attention_mask, position_ids)
    def hook_fn(module, args):
        nonlocal vector
        # we want activations, input[0]
        activation = args[0]
        vector = vector.to(activation)
        # steering - same number from vector is added to every token of every prompt, scaled by coeff
        activation += coeff * vector
        # returns (hidden_states, attention_mask, position_ids)
        return (activation, *args[1:])
    return hook_fn


def get_direction_ablation_input_pre_hook(direction):
    """adapted from hook_utils.py, removes the direction from the activations.
        Removes direction from actvations arriving at layer, the first point where 
        values enter the residual stream

        residual_stream = [residual_stream] + attention_output + mlp_out
    """
    def hook_fn(module, args):
        nonlocal direction
        activation = args[0]
        # normalise
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        # remove direction from activations
        activation -= (activation @ direction).unsqueeze(-1) * direction
        return (activation, *args[1:])
    return hook_fn


def get_direction_ablation_output_hook(direction):
    """adapted from hook_utils.py, removes direction from 
       sublayer's output, before added to residual stream. 
       Attached to attention and MLP.

        residual_stream = residual_stream + [attention_output] + [mlp_out]
    """
    def hook_fn(module, input, output):
        nonlocal direction
        activation = output[0] if isinstance(output, tuple) else output
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction
        return (activation, *output[1:]) if isinstance(output, tuple) else activation
    return hook_fn


def get_all_direction_ablation_hooks(model, direction):
    """taken from hook_utils.py, builds all 90 hooks,
       covering every point a value enters the residual stream
    """
    n = model.config.num_hidden_layers
    layers = model.model.layers
    pre = [(layers[l], get_direction_ablation_input_pre_hook(direction))
           for l in range(n)]
    fwd = [(layers[l].self_attn, get_direction_ablation_output_hook(direction))
           for l in range(n)]
    fwd += [(layers[l].mlp, get_direction_ablation_output_hook(direction))
            for l in range(n)]
    return pre, fwd


def get_mean_matching_pre_hook(direction, target_value):
    """Clamp each token's coordinate along direction to target_value at the input of a block.
    """
    def hook_fn(module, args):
        activation = args[0]
        d = direction.to(activation)
        current = activation @ d
        adjustment = (target_value - current).unsqueeze(-1) * d
        return (activation + adjustment, *args[1:])
    return hook_fn


def get_span_zero_hook(spans):
    """zero the v_proj output at the positions while reading the prompt."""
    def hook_fn(module, args, output):
        if output.shape[1] > 1:
            output = output.clone()
            for lo, hi in spans:
                output[:, lo:hi, :] = 0
        return output
    return hook_fn
