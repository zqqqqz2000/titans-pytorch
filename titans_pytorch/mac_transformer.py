from __future__ import annotations

from collections import namedtuple
from copy import deepcopy
from functools import partial
from math import ceil
from typing import Callable

import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, cat, nn, stack
from torch.nn import Linear, Module, ModuleList

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import (create_block_mask,
                                                   flex_attention)

    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    raise ImportError("You need to be on the latest pytorch with a cuda device available")


def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding=False):
    def create_mac_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len
        kv_without_mem = kv_idx - persist_mem_len
        causal_mask = q_idx >= kv_without_mem

        if not sliding:
            block_diagonal = (q_idx // window_size) == (kv_without_mem // window_size)
            causal_mask = causal_mask & block_diagonal
        else:
            sliding_mask = (q_idx - kv_without_mem) <= window_size
            causal_mask = causal_mask & sliding_mask

        return is_persist_mem | (~is_persist_mem & causal_mask)

    block_mask = create_block_mask(
        create_mac_mask,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len + persist_mem_len,
        _compile=True,
    )
    return block_mask


# einstein notation related

from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from hyper_connections import get_init_and_expand_reduce_stream_functions
from rotary_embedding_torch import RotaryEmbedding
from x_transformers.attend import Attend

from titans_pytorch.neural_memory import NeuralMemory

# b - batch
# n - sequence
# h - heads
# d - feature dimension

# absolute and relative positions


# hyper connections / attend from x-transformers, which handles different queries and key lengths better


# proposed neural memory


# constants

LinearNoBias = partial(Linear, bias=False)

AttnIntermediates = namedtuple("AttnIntermediates", ("value_residual", "cached_key_values"))

# helpers


def exists(v):
    return v is not None


def default[_T](v, d: _T) -> _T:
    return v if exists(v) else d


def identity(t):
    return t


def divisible_by(num, den):
    return (num % den) == 0


def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult


def round_down_multiple(seq, mult):
    return seq // mult * mult


def pack_with_inverse(t, pattern: str):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern: str | None = None):
        return unpack(out, packed_shape, default(inv_pattern, pattern))

    return packed, inverse


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def pad_and_segment_with_inverse(seq, segment_len, fold_into_batch=True, inverse_remove_pad=True):
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))

    if fold_into_batch:
        seq = rearrange(seq, "b (w n) d -> (b w) n d", n=segment_len)

    def inverse(out):

        if fold_into_batch:
            out = rearrange(out, "(b w) ... n d -> b ... (w n) d", b=batch)

        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]

        return out

    return seq, inverse


# sampling related


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0):
    if temperature > 0.0:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim=-1, keepdim=True)


# min_p
# https://arxiv.org/abs/2407.01082


def min_p_filter(logits, min_p=0.1):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float("-inf"), logits)


# feedforward and attention


class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim),
    )


class SegmentedAttention(Module):
    def __init__(
        self,
        dim,
        segment_len,
        num_persist_mem_tokens=0,
        num_longterm_mem_tokens=0,
        dim_head=64,
        heads=8,
        sliding=False,
        accept_value_residual=False,
        attend_kwargs: dict = dict(),
        use_flex_attn=False,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_inner = dim_head * heads

        self.rotary_emb = RotaryEmbedding(dim_head)

        self.attend = Attend(causal=True, **attend_kwargs)

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_learned_v_mix = (
            nn.Sequential(nn.Linear(dim, heads), Rearrange("b n h -> b h n 1"), nn.Sigmoid())
            if accept_value_residual
            else None
        )

        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens

        total_segment_len = segment_len + num_longterm_mem_tokens
        self.total_segment_len = total_segment_len

        self.sliding = sliding  # sliding window attn - doubt their non-sliding results being the best. local attention with overlapping windows is very strong

        self.split_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))

        # flex attn related

        assert not (
            use_flex_attn and not exists(flex_attention)
        ), "you need to be on the latest pytorch with a cuda device available"
        self.use_flex_attn = use_flex_attn

        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens

    def forward_inference(
        self,
        token,
        cache,
        value_residual=None,
        output_gating=None,
    ):
        batch = token.shape[0]

        # attention

        token = self.norm(token)

        q, k, v = self.to_qkv(token).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            assert self.to_learned_v_mix
            mix = self.to_learned_v_mix(token)
            v = v.lerp(value_residual, mix)

        # caching

        ck, cv = cache
        k = cat((ck, k), dim=-2)
        v = cat((cv, v), dim=-2)

        next_cache = (k, v)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold

        q, k, v = tuple(rearrange(t, "b h n d -> b h n d") for t in (q, k, v))

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, "kv ... -> kv b ...", b=k.shape[0])

        # persistent memory

        k = cat((pmk, k), dim=-2)
        v = cat((pmv, v), dim=-2)

        # attention

        out, _ = self.attend(q, k, v)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward_flex(
        self,
        seq,
        value_residual=None,
        flex_attn_fn: Callable | None = None,
        output_gating=None,
        cache=None,
    ):

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        batch, seq_len = seq.shape[:2]

        # attention

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            assert self.to_learned_v_mix
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # caching

        next_cache = (k, v)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, "kv h n d -> kv b h n d", b=batch)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # persistent memory

        k = cat((pmk, k), dim=-2)
        v = cat((pmv, v), dim=-2)

        # prep flex attention

        if not exists(flex_attn_fn):
            block_mask = create_mac_block_mask(
                seq_len,
                self.total_segment_len,
                self.num_persist_mem_tokens,
                self.sliding,
            )

            assert flex_attention
            flex_attn_fn = partial(flex_attention, block_mask=block_mask)

        # attention

        assert flex_attn_fn
        out = flex_attn_fn(q, k, v)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward(
        self,
        seq,
        value_residual=None,
        flex_attn_fn: Callable | None = None,
        disable_flex_attn=False,
        output_gating=None,
        cache=None,
    ):
        is_inferencing = exists(cache)

        if is_inferencing:
            assert seq.shape[-2] == 1
            return self.forward_inference(seq, cache, value_residual, output_gating=output_gating)

        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            return self.forward_flex(
                seq,
                value_residual,
                flex_attn_fn,
                output_gating=output_gating,
                cache=cache,
            )

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        segment_len, num_longterm_mem_tokens = (
            self.segment_len,
            self.num_longterm_mem_tokens,
        )
        total_segment_len = segment_len + num_longterm_mem_tokens

        batch, seq_len = seq.shape[:2]

        # auto pad to multiple

        seq, inverse_segment = pad_and_segment_with_inverse(seq, total_segment_len, fold_into_batch=False)

        # attention

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            assert self.to_learned_v_mix
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # caching

        next_cache = tuple(map(inverse_segment, (k, v)))

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold

        q, k, v = tuple(rearrange(t, "b h (w n) d -> (b w) h n d", n=total_segment_len) for t in (q, k, v))

        # maybe sliding for cpu

        attend_kwargs = dict()

        if self.sliding:
            k, v = tuple(rearrange(t, "(b w) ... -> b w ...", b=batch) for t in (k, v))
            k, v = tuple(pad_at_dim(t, (1, 0), value=0.0, dim=1) for t in (k, v))
            k = cat((k[:, :-1], k[:, 1:]), dim=-2)
            v = cat((v[:, :-1], v[:, 1:]), dim=-2)
            k, v = tuple(rearrange(t, "b w ... -> (b w) ...") for t in (k, v))

            # take care of masking

            idx = torch.arange(seq.shape[-2], device=seq.device)
            q_idx = rearrange(idx, "(w n) -> w n", n=total_segment_len)
            k_idx = pad_at_dim(q_idx, (1, 0), dim=0, value=-1e4)
            k_idx = cat((k_idx[:-1], k_idx[1:]), dim=-1)

            q_idx = rearrange(q_idx, "w i -> w i 1")
            k_idx = rearrange(k_idx, "w j -> w 1 j")

            sliding_mask = (q_idx - k_idx) <= total_segment_len
            sliding_mask = F.pad(sliding_mask, (self.num_persist_mem_tokens, 0), value=True)

            sliding_mask = repeat(sliding_mask, "w i j -> (b w) 1 i j", b=batch)
            attend_kwargs.update(mask=sliding_mask)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, "kv ... -> kv b ...", b=k.shape[0])

        # persistent memory

        k = cat((pmk, k), dim=-2)
        v = cat((pmv, v), dim=-2)

        # attention

        out, _ = self.attend(q, k, v, **attend_kwargs)

        out = self.merge_heads(out)

        out = self.to_out(out)

        out = rearrange(out, "(b w) n d -> b (w n) d", b=batch)

        out = inverse_segment(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)


# MAC transformer


class MemoryAsContextTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        segment_len,
        neural_memory_segment_len=None,
        neural_mem_gate_attn_output=False,
        neural_memory_add_value_residual=False,
        num_longterm_mem_tokens=0,
        num_persist_mem_tokens=0,
        dim_head=64,
        heads=8,
        ff_mult=4,
        num_residual_streams=4,
        neural_memory_model: Module | None = None,
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        aux_kv_recon_loss_weight=0.0,
        use_flex_attn=False,
        sliding_window_attn=False,
        weight_tie_memory_model=False,
        prev_neural_mem_update_for_weights=None,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim=dim, num_axial_dims=2)

        # long term mem tokens

        self.segment_len = segment_len

        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        has_longterm_mems = num_longterm_mem_tokens > 0

        self.longterm_mems = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim) * 0.02)

        # maybe sliding window attn

        self.sliding_window_attn = sliding_window_attn
        self.attn_window_size = segment_len + num_longterm_mem_tokens

        # hyper conection

        (
            init_hyper_conn,
            self.expand_streams,
            self.reduce_streams,
        ) = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable=num_residual_streams == 1)

        self.layers = ModuleList([])

        self.neural_memory_segment_len = default(neural_memory_segment_len, num_longterm_mem_tokens + segment_len)

        layers = tuple(range(1, depth + 1))

        neural_memory_layers = default(neural_memory_layers, layers)

        # weight tying neural memory model

        maybe_copy = deepcopy if not weight_tie_memory_model else identity

        if weight_tie_memory_model:
            assert exists(neural_memory_model), "`neural_memory_model` must be explicitly set"

        self.weight_tie_memory_model = weight_tie_memory_model
        self.prev_neural_mem_update_for_weights = default(prev_neural_mem_update_for_weights, weight_tie_memory_model)

        # value residual learning for neural memory

        is_first_mem = True
        self.mem_add_value_residual = neural_memory_add_value_residual

        # mem, attn, and feedforward layers

        for layer in layers:
            is_first = layer == 1

            # attention and feedforward

            attn = SegmentedAttention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                segment_len=segment_len,
                use_flex_attn=use_flex_attn,
                accept_value_residual=not is_first,
                num_longterm_mem_tokens=num_longterm_mem_tokens,
                num_persist_mem_tokens=num_persist_mem_tokens,
                sliding=sliding_window_attn,
            )

            mem = None
            mem_hyper_conn = None

            if layer in neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(dim=dim, add_branch_out_to_residual=not neural_mem_gate_attn_output)

                mem = NeuralMemory(
                    dim=dim,
                    chunk_size=self.neural_memory_segment_len,
                    model=maybe_copy(neural_memory_model),
                    accept_value_residual=not is_first_mem and neural_memory_add_value_residual,
                    **neural_memory_kwargs,
                )

                is_first_mem = False

            ff = FeedForward(dim=dim, mult=ff_mult)

            self.layers.append(
                ModuleList(
                    filter(
                        None,
                        [
                            mem_hyper_conn,
                            mem,
                            init_hyper_conn(dim=dim, branch=attn),
                            init_hyper_conn(dim=dim, branch=ff),
                        ],
                    )
                )
            )

        self.norm = nn.RMSNorm(dim)

        self.to_logits = LinearNoBias(dim, num_tokens)

        # whether to gate the attention output with the retrieved memories

        self.gate_attn_output = neural_mem_gate_attn_output

        # auxiliary loss on kv recon

        self.has_aux_kv_recon_loss = aux_kv_recon_loss_weight > 0.0
        self.aux_kv_recon_loss_weight = aux_kv_recon_loss_weight

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # flex attn related

        assert not (
            use_flex_attn and not exists(flex_attention)
        ), "you need to be on the latest pytorch with a cuda device available"
        self.use_flex_attn = use_flex_attn

        self.num_persist_mem_tokens = num_persist_mem_tokens

    def seq_index_is_longterm(self, seq_index):
        total_segment_len, segment_len = self.attn_window_size, self.segment_len
        return ((seq_index % total_segment_len + 1) - segment_len) > 0

    def seq_len_with_longterm_mem(self, seq_len):
        assert seq_len > 0

        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return ((seq_len - 1) // segment_len) * num_mem + seq_len

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature=1.5,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(
            min_p=0.1,
        ),
        show_progress=True,
        use_cache=False,
    ):
        was_training = self.training
        self.eval()

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        # cache for axial pos, attention, and neural memory

        cache = None
        factorized_pos_emb = None

        # precompute factorized pos emb

        if use_cache:
            seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)

            axial_dims = self.axial_pos_emb.maybe_derive_outer_dim(seq_len_with_mem, (self.neural_memory_segment_len,))

            factorized_pos_emb = self.axial_pos_emb(axial_dims, return_factorized=True)

        # sample

        with tqdm.tqdm(total=sample_num_times, disable=not show_progress) as pbar:

            while out.shape[-1] < seq_len:

                logits, next_cache = self.forward(
                    out,
                    disable_flex_attn=True,
                    cache=cache,
                    return_cache=True,
                    factorized_pos_emb=factorized_pos_emb,
                )

                if use_cache:
                    cache = next_cache

                if not exists(logits):
                    continue

                logits = logits[:, -1]

                logits = filter_fn(logits, **filter_kwargs)
                sample = gumbel_sample(logits, temperature=temperature)

                out = torch.cat((out, sample), dim=-1)
                pbar.update(1)

        self.train(was_training)

        return out[..., prompt_seq_len:]

    def forward(
        self,
        x,
        return_loss=False,
        return_loss_breakdown=False,
        disable_flex_attn=False,
        cache=None,
        return_cache=False,
        factorized_pos_emb=None,
    ):

        labels = None
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # math

        (
            batch,
            seq_len,
            neural_mem_segment_len,
            segment_len,
            num_longterm_mem_tokens,
            attn_window_size,
            prev_neural_mem_update_for_weights,
        ) = (
            *x.shape,
            self.neural_memory_segment_len,
            self.segment_len,
            self.num_longterm_mem_tokens,
            self.attn_window_size,
            self.prev_neural_mem_update_for_weights,
        )

        seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)

        # token embedding

        x = self.token_emb(x)

        # intersperse longterm memory

        x, inverse_segment = pad_and_segment_with_inverse(x, segment_len, inverse_remove_pad=False)

        mems = repeat(self.longterm_mems, "n d -> b n d", b=x.shape[0])
        x, inverse_pack_mems = pack_with_inverse((x, mems), "b * d")

        x = inverse_segment(x)

        # splice out unneeded tokens from padding for longterm mems

        x = x[:, :seq_len_with_mem]

        # apply axial positional embedding
        # so intra and inter segment can be more easily discerned by the network

        pos_emb = self.axial_pos_emb.forward_with_seq_len(
            seq_len_with_mem, (neural_mem_segment_len,), factorized=factorized_pos_emb
        )

        x = x + pos_emb

        # prep flex attention

        use_flex_attn = x.is_cuda and self.use_flex_attn and not disable_flex_attn

        flex_attn_fn = None

        if use_flex_attn:
            block_mask = create_mac_block_mask(
                seq_len_with_mem,
                self.attn_window_size,
                self.num_persist_mem_tokens,
                self.sliding_window_attn,
            )
            assert flex_attention
            flex_attn_fn = partial(flex_attention, block_mask=block_mask)

        # kv caching

        is_inferencing = exists(cache)

        if not exists(cache):
            cache = (seq_len_with_mem - 1, None, None)

        assert cache
        inference_seq_index, kv_caches, neural_mem_caches = cache

        kv_caches = iter(default(kv_caches, []))
        neural_mem_caches = iter(default(neural_mem_caches, []))

        next_kv_caches = []
        next_neural_mem_caches = []

        # weight tied neural memory

        neural_memory_updates = None

        # value residual

        value_residual = None

        mem_value_residual = None

        # aux losses

        kv_recon_losses = self.zero

        # when inferencing, only do one token at a time

        if is_inferencing:
            ind = inference_seq_index
            x = x[:, ind : (ind + 1)]

        # expand and reduce streams for hyper connections

        x = self.expand_streams(x)

        for mem_hyper_conn, mem, attn, ff in self.layers:  # type: ignore

            retrieved = None
            attn_out_gates = None
            next_neural_mem_cache = None

            # maybe neural memory

            if exists(mem):

                mem_input, add_residual = mem_hyper_conn(x)

                if not is_inferencing:
                    (
                        retrieved,
                        next_neural_mem_cache,
                        next_mem_value_residual,
                    ), mem_kv_aux_loss = mem(
                        mem_input,
                        return_aux_kv_loss=True,
                        return_values=True,
                        value_residual=mem_value_residual,
                        prev_layer_updates=neural_memory_updates,
                    )

                    kv_recon_losses = kv_recon_losses + mem_kv_aux_loss

                else:
                    (
                        retrieved,
                        next_neural_mem_cache,
                        next_mem_value_residual,
                    ) = mem.forward_inference(
                        mem_input,
                        state=next(neural_mem_caches, None),
                        return_values=True,
                        value_residual=mem_value_residual,
                        prev_layer_updates=neural_memory_updates,
                    )

                if self.mem_add_value_residual:
                    mem_value_residual = next_mem_value_residual

                if prev_neural_mem_update_for_weights:
                    neural_memory_updates = next_neural_mem_cache.updates

                if self.gate_attn_output:
                    attn_out_gates = retrieved.sigmoid()
                else:
                    x = add_residual(retrieved)

            # attention

            x, (values, next_kv_cache) = attn(
                x,
                value_residual=value_residual,
                disable_flex_attn=disable_flex_attn,
                flex_attn_fn=flex_attn_fn,
                output_gating=attn_out_gates,
                cache=next(kv_caches, None),
            )

            value_residual = default(value_residual, values)

            # caches

            next_kv_caches.append(next_kv_cache)
            next_neural_mem_caches.append(next_neural_mem_cache)

            # feedforward

            x = ff(x)

        # taking care of cache first
        # for early return when processing long term mem tokens during inference

        next_cache = None
        if return_cache:
            next_kv_caches = stack([stack(kv_cache) for kv_cache in next_kv_caches])

            # handle kv cache length depending on local attention type

            next_kv_caches = next_kv_caches[..., -attn_window_size:, :]

            kv_cache_length = next_kv_caches.shape[-2]

            if not self.sliding_window_attn and divisible_by(kv_cache_length, attn_window_size):
                next_kv_caches = next_kv_caches[..., 0:0, :]

            next_cache = (
                inference_seq_index + 1,
                next_kv_caches,
                next_neural_mem_caches,
            )

            is_longterm_mem = self.seq_index_is_longterm(inference_seq_index)

            if is_inferencing and is_longterm_mem:
                return None, next_cache

        # hyper connection reducing of streams

        x = self.reduce_streams(x)

        # excise out the memories

        if not is_inferencing:

            x, inverse_segment = pad_and_segment_with_inverse(x, attn_window_size, inverse_remove_pad=False)

            x, _ = inverse_pack_mems(x)

            x = inverse_segment(x)

            x = x[:, :seq_len]

        # to logits

        x = self.norm(x)

        logits = self.to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, next_cache

        assert labels
        ar_loss = F.cross_entropy(rearrange(logits, "b n l -> b l n"), labels)

        losses = ar_loss

        if self.has_aux_kv_recon_loss:
            losses = losses + kv_recon_losses * self.aux_kv_recon_loss_weight

        if not return_loss_breakdown:
            return losses

        return losses, (ar_loss, kv_recon_losses)
