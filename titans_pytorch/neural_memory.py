from __future__ import annotations
from typing import Callable

import math
from functools import partial
from collections import namedtuple

import torch
from torch import nn, cat, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter, ParameterList
from torch.func import functional_call, vmap, grad

from tensordict import TensorDict

from titans_pytorch.associative_scan import (
    associative_scan,
    binary_operator,
    pad_at_dim,
)

from titans_pytorch.memory_models import MemoryMLP

import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
n - sequence
d - feature dimension
c - intra-chunk
w - num memory network weight parameters
"""

LinearNoBias = partial(Linear, bias=False)

NeuralMemCache = namedtuple(
    "NeuralMemCache", ["seq", "cache_store_segment", "states", "updates"]
)

# functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def xnor(x, y):
    return not (x ^ y)


def safe_cat(inputs, dim=-2):
    inputs = tuple(filter(exists, inputs))

    if len(inputs) == 0:
        return None
    elif len(inputs) == 1:
        return inputs[0]

    return cat(inputs, dim=dim)


def identity(t):
    return t


def pair(v):
    return (v, v) if not isinstance(v, tuple) else v


def round_down_multiple(seq, mult):
    return seq // mult * mult


def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult


def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse


def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    if len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)


# softclamping gradients


def softclamp_max(t, max_value):
    half_max_value = max_value / 2
    return ((t / half_max_value).tanh() * half_max_value) + half_max_value


def softclamp_grad_norm(t, max_value):
    t, inverse = pack_one_with_inverse(t, "bn *")

    norm = t.norm(dim=-1, keepdim=True)
    clamped_norm = softclamp_max(norm, max_value)

    t = t * (clamped_norm / norm)
    return inverse(t)


# multi head rmsnorm


class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine=False)
        self.gamma = Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.0)


# chunk pooling


class AveragePool(Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, x, chunk_size=None):
        chunk_size = default(chunk_size, self.chunk_size)
        return reduce(x, "b (n c) d -> b n d", "mean", c=chunk_size)


class AttentionPool(Module):
    def __init__(self, dim, chunk_size):
        """
        taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.to_attn_logits = nn.Linear(dim, dim)

        # default to average pool

        nn.init.zeros_(self.to_attn_logits.weight)
        nn.init.zeros_(self.to_attn_logits.bias)

    def forward(self, x, chunk_size=None):
        chunk_size = default(chunk_size, self.chunk_size)

        x = rearrange(x, "b (n c) d -> b n c d", c=chunk_size)

        attn_logits = self.to_attn_logits(x)

        attn = attn_logits.softmax(dim=-2)

        return reduce(x * attn, "b n c d -> b n d", "sum")


# associative scan wrapper


class AssocScan(Module):
    def __init__(self, use_accelerated=False):
        super().__init__()
        self.use_accelerated = use_accelerated

    def forward(self, gates, inputs, prev=None, remove_prev=None):
        remove_prev = default(remove_prev, exists(prev))

        if exists(prev):
            inputs, _ = pack([prev, inputs], "b * d")
            gates = pad_at_dim(gates, (1, 0), value=1.0, dim=-2)

        if not self.use_accelerated:
            _, out = associative_scan(binary_operator, (gates, inputs))

            if remove_prev:
                out = out[:, 1:]

            return out

        from accelerated_scan.triton import scan as triton_scan
        from accelerated_scan.warp import scan as warp_scan

        scan = triton_scan if gates.is_cuda else warp_scan

        def accelerate_scan_fn(gates, inputs):
            gates = gates.expand_as(inputs)
            gates, inputs = tuple(
                rearrange(t, "b n d -> b d n") for t in (gates, inputs)
            )

            seq_len = gates.shape[-1]
            next_power_two_seq_len = 2 ** max(5, int(math.ceil(math.log2(seq_len))))

            gates = F.pad(gates, (0, next_power_two_seq_len - seq_len))
            inputs = F.pad(inputs, (0, next_power_two_seq_len - seq_len))

            outputs = scan(gates.contiguous(), inputs.contiguous())

            outputs = outputs[..., :seq_len]
            outputs = rearrange(outputs, "b d n -> b n d")
            return outputs

        out = accelerate_scan_fn(gates, inputs)

        if remove_prev:
            out = out[:, 1:]

        return out


# main neural memory


def default_adaptive_step_transform(adaptive_step, max_lr=1e-2):
    return adaptive_step.sigmoid() * max_lr


def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim=-1)


class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        chunk_size: int | tuple[int, int] = 1,
        dim_head=None,
        heads=1,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr=1e-2,
        per_parameter_lr_modulation=False,  # allow outer network to control learning rate per weight matrix of memory network
        max_mem_layer_modulation=1e1,  # max of 10.
        attn_pool_chunks=False,
        momentum=True,
        pre_rmsnorm=True,
        post_rmsnorm=True,
        qk_rmsnorm=False,
        accept_value_residual=False,
        max_grad_norm: float | None = None,
        use_accelerated_scan=False,
        activation: Module | None = None,
        default_model_kwargs: dict = dict(depth=2),
    ):
        super().__init__()
        dim_head = default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)

        self.retrieve_chunk_size, self.store_chunk_size = pair(chunk_size)

        # associative scan

        self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)

        # norms

        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.multihead_rmsnorm = (
            MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()
        )

        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()

        # maybe multi-headed

        dim_inner = dim_head * heads

        self.heads = heads

        self.split_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")
        self.combine_heads = (
            LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()
        )

        self.retrieve_gate = (
            Sequential(
                LinearNoBias(dim, heads), Rearrange("b n h -> b h n 1"), nn.Sigmoid()
            )
            if heads > 1
            else None
        )

        # memory mlp

        if not exists(model):
            model = MemoryMLP(dim_head, **default_model_kwargs)

        assert not exists(
            next(model.buffers(), None)
        ), "model cannot have buffers for now"

        # the memory is the weights of the model

        self.memory_model = model

        self.num_memory_parameter_tensors = len(set(model.parameters()))

        # the chunk size within the paper where adaptive step, momentum, weight decay are shared

        self.chunk_size = chunk_size

        # prepare function for per sample gradients from model above, using torch.func

        def forward_and_loss(params, inputs, loss_weights, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(
                pred, target
            )  # simple mse loss in paper - eq (12) - |M(k) - v|²
            weighted_loss = loss * loss_weights
            return weighted_loss.sum(), weighted_loss.mean()

        # two functions

        grad_fn = grad(forward_and_loss, has_aux=True)

        self.per_sample_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0, 0))
        self.per_sample_grad_fn_expanded_weights = vmap(grad_fn, in_dims=(0,) * 4)

        # queries for retrieving from the model

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), activation)

        # keys and values for storing to the model

        self.to_keys_values = Sequential(LinearNoBias(dim, dim_inner * 2), activation)
        self.store_memory_loss_fn = store_memory_loss_fn

        # value residual learning

        self.learned_value_residual = (
            Sequential(
                LinearNoBias(dim, heads), Rearrange("b n h -> b h n 1"), nn.Sigmoid()
            )
            if accept_value_residual
            else None
        )

        # empty memory embed

        self.empty_memory_embed = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.empty_memory_embed, std=0.02)

        # `chunk_size` refers to chunk size used for storing to memory model weights

        chunk_size = self.store_chunk_size

        # whether to use averaging of chunks, or attention pooling

        assert not (
            attn_pool_chunks and chunk_size == 1
        ), "`attn_pool_chunks` cannot be set to True if `chunk_size` is set to 1"

        if not attn_pool_chunks:
            self.reduce_to_chunk_rep = AveragePool(chunk_size=chunk_size)
        else:
            self.reduce_to_chunk_rep = AttentionPool(dim, chunk_size=chunk_size)

        # learned adaptive learning rate and momentum

        self.to_momentum = (
            Sequential(LinearNoBias(dim, heads), Rearrange("b n h -> (b h) n 1"))
            if momentum
            else None
        )

        self.to_adaptive_step = Sequential(
            LinearNoBias(dim, heads), Rearrange("b n h -> (b h) n")
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(
                default_adaptive_step_transform, max_lr=default_step_transform_max_lr
            )

        self.adaptive_step_transform = adaptive_step_transform

        # per layer learning rate modulation

        self.to_layer_modulation = (
            Sequential(
                LinearNoBias(dim, heads * self.num_memory_parameter_tensors),
                Rearrange("b n (h w) -> w (b h) n", h=heads),
                nn.Sigmoid(),
            )
            if per_parameter_lr_modulation
            else None
        )

        self.max_mem_layer_modulation = max_mem_layer_modulation

        # allow for softclamp the gradient norms for storing memories

        self.max_grad_norm = max_grad_norm

        # weight decay factor

        self.to_decay_factor = Sequential(
            LinearNoBias(dim, heads), Rearrange("b n h -> (b h) n 1")
        )

        # maybe use accelerated scan

        self.use_accelerated_scan = use_accelerated_scan

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    def init_weights(self):
        weights = TensorDict(dict(self.memory_model.named_parameters()))
        return weights

    def init_empty_memory_embed(self, batch, seq_len):
        return repeat(self.empty_memory_embed, "d -> b n d", b=batch, n=seq_len)

    def store_memories(
        self,
        seq,
        weights: dict[str, Tensor],
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        prev_layer_updates: dict[str, Tensor] | None = None,
        return_aux_kv_loss=False,
        chunk_size=None,
        value_residual=None,
    ):
        assert xnor(exists(value_residual), exists(self.learned_value_residual))

        seq_len, heads, chunk_size = (
            seq.shape[-2],
            self.heads,
            default(chunk_size, self.store_chunk_size),
        )

        # handle edge case

        if seq_len < chunk_size:
            return TensorDict(weights).clone().zero_(), self.zero

        seq = self.store_norm(seq)

        # curtail sequence by multiple of the chunk size
        # only a complete chunk of the sequence provides the memory for the next chunk

        round_down_seq_len = round_down_multiple(seq_len, chunk_size)

        seq = seq[:, :round_down_seq_len]

        # per sample grad function

        per_sample_grad_fn = self.per_sample_grad_fn

        # weights of the memory network

        weights = TensorDict(weights)

        # allow for neural memory of a previous layer and the past to produce gradients that become the weights of the current one generating the surprise
        # think this is necessary otherwise the memory model is static (unless if paper is misunderstood)
        # improvise (or perhaps correcting to) a solution

        if exists(prev_layer_updates):
            prev_layer_updates = TensorDict(prev_layer_updates)

            weights = weights + prev_layer_updates

            per_sample_grad_fn = (
                self.per_sample_grad_fn_expanded_weights
            )  # the weights will now have a batch * chunk dimension

        # derive learned hparams for optimization of memory network

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size=chunk_size)

        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        need_layer_lr_mod = exists(self.to_layer_modulation)
        has_momentum = exists(self.to_momentum)

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()

        if need_layer_lr_mod:
            layer_lr_mod = (
                self.to_layer_modulation(chunked_seq) * self.max_mem_layer_modulation
            )

        # keys and values

        keys, values = self.to_keys_values(seq).chunk(2, dim=-1)

        # maybe multi head

        keys, values = map(self.split_heads, (keys, values))

        batch = keys.shape[0]

        # maybe qk rmsnorm

        keys = self.k_norm(keys)

        # maybe value residual learning

        orig_values = values

        if exists(self.learned_value_residual):
            mix = self.learned_value_residual(seq)
            values = values.lerp(value_residual, mix)

        # take care of chunking

        keys, values = tuple(
            rearrange(t, "b h (n c) d -> (b h n) c d", c=chunk_size)
            for t in (keys, values)
        )

        adaptive_lr = rearrange(adaptive_lr, "b (n c) -> (b n) c", c=chunk_size)

        # flatten batch and time if surprise depends on previous layer memory model

        if exists(prev_layer_updates):
            weights = weights.apply(lambda t: rearrange(t, "b n ... -> (b n) ..."))

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)

        grads, aux_kv_recon_loss = per_sample_grad_fn(
            dict(weights), keys, adaptive_lr, values
        )

        grads = TensorDict(grads)

        # maybe softclamp grad norm

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        # restore batch and sequence dimension

        grads = grads.apply(
            lambda t: rearrange(t, "(b n) ... -> b n ...", b=batch * heads)
        )

        # maybe per layer modulation

        if need_layer_lr_mod:
            grads = TensorDict(
                {
                    name: einx.multiply("b h, b h ... -> b h ...", layer_lr_mod, t)
                    for layer_lr_mod, (name, t) in zip(layer_lr_mod, grads.items())
                }
            )

        # negative gradients, adaptive lr already applied as loss weight

        surprises = grads.apply(lambda t: -t)

        # past states

        if not exists(past_state):
            empty_dict = {key: None for key in weights.keys()}
            past_state = (empty_dict, empty_dict)

        past_last_update, past_last_momentum = past_state

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates

        next_momentum = TensorDict() if has_momentum else None
        updates = TensorDict()

        next_last_update = TensorDict()
        next_last_momentum = TensorDict()

        for (param_name, surprise), (_, last_update), (_, last_momentum) in zip(
            surprises.items(), past_last_update.items(), past_last_momentum.items()
        ):

            surprise, inverse_pack = pack_one_with_inverse(surprise, "b n *")

            update = surprise

            # derive momentum with associative scan - eq (10)

            if has_momentum:
                update = self.assoc_scan(
                    adaptive_momentum, surprise, prev=last_momentum
                )  # momentum is S / surprise in the paper
                momentum = update
                next_last_momentum[param_name] = momentum[:, -1]

            # use associative scan again for learned forgetting (weight decay) - eq (13)

            update = self.assoc_scan(1.0 - decay_factor, update, prev=last_update)
            next_last_update[param_name] = update[:, -1]

            updates[param_name] = inverse_pack(update)

            if has_momentum:
                next_momentum[param_name] = inverse_pack(momentum)

        # compute next states for inference, or titans-xl like training

        next_state = (next_last_update, next_last_momentum)

        # returns

        output = (updates, next_state, orig_values)

        if not return_aux_kv_loss:
            return output

        return output, aux_kv_recon_loss.mean()

    def retrieve_memories(
        self,
        seq,
        past_weights: dict[str, Tensor],
        chunk_size=None,
        prev_layer_updates: dict[str, Tensor] | None = None,
    ):
        chunk_size = default(chunk_size, self.retrieve_chunk_size)
        batch, seq_len = seq.shape[:2]

        seq = self.retrieve_norm(seq)

        if seq_len < chunk_size:
            return self.init_empty_memory_embed(batch, seq_len)

        seq = seq[:, (chunk_size - 1) :]
        curtailed_seq_len = seq.shape[-2]

        next_seq_len = round_up_multiple(curtailed_seq_len, chunk_size)

        padding = next_seq_len - curtailed_seq_len
        seq = pad_at_dim(seq, (0, padding), dim=1)

        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature (recall fetching of memories is q @ (kv)) / schmidhuber's paper

        curr_weights = TensorDict(past_weights)

        if exists(prev_layer_updates):
            curr_weights = curr_weights + TensorDict(prev_layer_updates)

        # sequence Float['b n d'] to queries

        queries = self.to_queries(seq)

        # maybe multihead

        queries = self.split_heads(queries)

        # maybe qk rmsnorm

        queries = self.q_norm(queries)

        # fetch values from memory model

        curr_weights = curr_weights.apply(
            lambda t: rearrange(t, "b n ... -> (b n) ...")
        )
        queries = rearrange(queries, "b h (n c) d -> (b h n) c d", c=chunk_size)

        # forward functional call

        values = functional_call(self.memory_model, dict(curr_weights), queries)

        # reconstitute batch dimension

        values = rearrange(values, "(b h n) c d -> b h (n c) d", b=batch, h=self.heads)

        values = self.multihead_rmsnorm(values)

        # maybe gate

        if exists(self.retrieve_gate):
            values = values * self.retrieve_gate(seq)

        # maybe merge heads and combine

        values = self.merge_heads(values)

        values = self.combine_heads(values)

        # restore, pad with empty memory embed

        empty_memory_embeds = self.init_empty_memory_embed(
            values.shape[0], chunk_size - 1
        )
        values = torch.cat((empty_memory_embeds, values), dim=-2)

        return values[:, :seq_len]

    @torch.no_grad()
    def forward_inference(
        self,
        token: Tensor,
        state=None,
        prev_layer_updates: dict[str, Tensor] | None = None,
        return_values=False,
        value_residual=None,
    ):

        # unpack previous state

        if not exists(state):
            state = (0, None, None, None)

        seq_index, cache_store_seq, past_states, updates = state

        curr_seq_len = seq_index + 1
        batch = token.shape[0]

        if token.ndim == 2:
            token = rearrange(token, "b d -> b 1 d")

        # get memory model weights

        weights = self.init_weights()

        # increment the sequence cache which is at most the chunk size

        cache_store_seq = safe_cat((cache_store_seq, token), dim=-2)

        # early return empty memory, when no memories are stored for steps < first chunk size

        if curr_seq_len < self.chunk_size:
            empty_mem = self.init_empty_memory_embed(batch, 1)

            output = empty_mem, NeuralMemCache(
                curr_seq_len, cache_store_seq, past_states, updates
            )

            if return_values:
                output = (*output, self.zero)

            return output

        # store if storage sequence cache hits the chunk size

        next_states = past_states
        store_seq_cache_len = cache_store_seq.shape[-2]

        if not exists(updates):
            updates = weights.clone().zero_()
            updates = updates.apply(lambda t: repeat(t, "... -> b 1 ...", b=batch))
        else:
            updates = updates.apply(lambda t: t[:, -1:])

        if exists(prev_layer_updates):
            prev_layer_updates = TensorDict(prev_layer_updates)
            prev_layer_updates = prev_layer_updates.apply(lambda t: t[:, -1:])

        values = None

        if store_seq_cache_len == self.chunk_size:

            next_updates, next_states, values = self.store_memories(
                cache_store_seq,
                weights,
                past_state=past_states,
                prev_layer_updates=prev_layer_updates,
                value_residual=value_residual,
            )

            updates = next_updates
            cache_store_seq = None

        # retrieve

        retrieved = self.retrieve_memories(token, updates + weights, chunk_size=1)

        # next state tuple

        next_state = NeuralMemCache(curr_seq_len, cache_store_seq, next_states, updates)

        output = (retrieved, next_state)

        if return_values:
            output = (*output, values)

        return output

    def forward(
        self,
        seq,
        store_seq=None,
        mem_model_weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        return_aux_kv_loss=False,
        chunk_size=None,
        store_chunk_size=None,
        return_values=False,
        value_residual=None,
        return_next_state=False,
        prev_layer_updates: dict[str, Tensor] | None = None,
    ):
        batch, seq_len = seq.shape[:2]

        if seq_len < self.retrieve_chunk_size:
            out = self.init_empty_memory_embed(batch, seq_len)

            next_store_state = NeuralMemCache(seq_len, seq, None, None)

            out = (out, next_store_state)

            if return_values:
                out = (*out, self.zero)

            if not return_aux_kv_loss:
                return out

            return out, self.zero

        if not exists(mem_model_weights):
            mem_model_weights = self.init_weights()

        # store

        store_seq = default(store_seq, seq)

        store_seq_len = store_seq.shape[-2]
        store_chunk_size = default(store_chunk_size, chunk_size, self.store_chunk_size)
        remainder = store_seq_len % store_chunk_size

        (updates, next_state, values), aux_kv_recon_loss = self.store_memories(
            store_seq,
            mem_model_weights,
            chunk_size=store_chunk_size,
            prev_layer_updates=prev_layer_updates,
            value_residual=value_residual,
            return_aux_kv_loss=True,
        )

        # retrieve

        retrieved = self.retrieve_memories(
            seq,
            mem_model_weights + updates,
            chunk_size=chunk_size,
            prev_layer_updates=prev_layer_updates,
        )

        # determine state for the storing of memories
        # for transformer-xl like training with neural memory as well as inferencing with initial prompt

        cache_store_seq = None

        if remainder > 0:
            cache_store_seq = store_seq[:, -remainder:]

        next_store_state = NeuralMemCache(seq_len, cache_store_seq, next_state, updates)

        output = (retrieved, next_store_state)

        if return_values:
            output = (*output, values)

        if not return_aux_kv_loss:
            return output

        return output, aux_kv_recon_loss
