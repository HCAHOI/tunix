# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OPT (Open Pre-trained Transformer) decoder-only language model."""

from collections.abc import Callable
import dataclasses
import enum
import functools

from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
import jaxtyping
from tunix.utils import compat
from tunix.utils import env_utils

K_MASK = -2.3819763e38

# OPT positional embeddings are shifted by 2 (fairseq padding_idx convention).
_OPT_POS_OFFSET = 2

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


env_utils.setup_sharding_environment()


class RematConfig(enum.Enum):
  """Recomputation scope: none, attention/FFN blocks, or a whole layer."""

  NONE = enum.auto()
  BLOCK = enum.auto()
  DECODER = enum.auto()


@dataclasses.dataclass(frozen=True, slots=True)
class ShardingConfig:
  """Parameter partition specs for the OPT model."""

  emb_vd: P
  emb_dv: P
  column_weight: P
  row_weight: P
  norm_weight: P
  score_weight_d1: P

  @staticmethod
  def get_default_sharding(is_sampling: bool = False) -> "ShardingConfig":
    """Returns TP partition specs, adding FSDP when not sampling."""
    fsdp = None if is_sampling else "fsdp"
    # Shard embedding width so vocabulary sizes need not be divisible by TP.
    embedding_width = (fsdp, "tp") if fsdp else "tp"
    return ShardingConfig(
        emb_vd=P(None, embedding_width),
        emb_dv=P(embedding_width, None),
        column_weight=P(fsdp, "tp"),
        row_weight=P("tp", fsdp),
        norm_weight=P("tp"),
        score_weight_d1=P(fsdp, None),
    )


def _get_activation(name: str) -> Callable[[jax.Array], jax.Array]:
  """Returns the JAX activation corresponding to a supported HF name."""
  if name == "relu":
    return jax.nn.relu
  if name in ("silu", "swish"):
    return jax.nn.silu
  if name == "gelu":
    return functools.partial(jax.nn.gelu, approximate=False)
  if name in ("gelu_new", "gelu_fast"):
    return functools.partial(jax.nn.gelu, approximate=True)
  raise ValueError(
      f"Unsupported activation: {name}. Expected one of: "
      "relu, silu, swish, gelu, gelu_new, gelu_fast."
  )


def _attention_mask(tokens, cache, mask, segment_ids):
  """Builds a causal mask in cache-slot coordinates, including packed segments.

  Positional embeddings may reset at padding or packed-sequence boundaries;
  local attention and cache writes must instead use physical cache slots.

  Args:
    tokens: Token IDs with shape [batch, query].
    cache: Per-layer cache, or None for an uncached forward pass.
    mask: Optional mask broadcastable to [batch, query, key]. Nonzero entries
      allow attention to the corresponding keys.
    segment_ids: Optional [batch, query] document IDs; zero marks padding.

  Returns:
    A tuple containing the causal attention mask, updated key segment IDs (or
    None), and physical query indices. The input cache is not mutated.
  """
  batch, length = tokens.shape
  first = cache["layer_0"] if cache is not None else None
  start = first["end_index"][0] if first is not None else 0
  key_length = first["k"].shape[1] if first is not None else length
  queries = start + jnp.arange(length)
  causal = jnp.arange(key_length)[None, :] <= queries[:, None]
  mask = causal[None, :, :] if mask is None else mask.astype(jnp.bool_) & causal
  key_segments = None
  if segment_ids is not None:
    if first is None:
      key_segments = segment_ids
    else:
      previous = first.get(
          "segment_ids", jnp.zeros((batch, key_length), jnp.int32)
      )
      key_segments = jax.lax.dynamic_update_slice(
          previous, segment_ids, (0, start)
      )
    same_segment = segment_ids[:, :, None] == key_segments[:, None, :]
    mask = mask & same_segment & (segment_ids[:, :, None] != 0)
  return mask, key_segments, queries


@dataclasses.dataclass(slots=True)
class ModelConfig:
  """Configuration for the OPT model."""

  num_layers: int
  vocab_size: int
  embed_dim: int  # hidden_size
  word_embed_proj_dim: int  # embedding width (may differ from hidden_size)
  hidden_dim: int  # ffn_dim
  num_heads: int
  head_dim: int
  max_position_embeddings: int
  norm_eps: float
  do_layer_norm_before: bool = False
  hidden_act: str = "relu"
  use_tied_embedding: bool = True
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE

  @property
  def num_kv_heads(self) -> int:  # OPT is full MHA.
    """Number of key/value heads for multi-head attention."""
    return self.num_heads

  @property
  def has_projection(self) -> bool:
    """Whether embeddings need a projection to the hidden dimension."""
    return self.word_embed_proj_dim != self.embed_dim

  @classmethod
  def opt_350m(cls):
    """Returns OPT-350M with post-LN and 512-dimensional token embeddings."""
    return cls(
        num_layers=24,
        vocab_size=50272,
        embed_dim=1024,
        word_embed_proj_dim=512,
        hidden_dim=4096,
        num_heads=16,
        head_dim=64,
        max_position_embeddings=2048,
        norm_eps=1e-05,
        do_layer_norm_before=False,
        hidden_act="relu",
    )


class LayerNorm(nnx.Module):
  """Standard LayerNorm (weight + bias), computed in float32."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float,
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.w = nnx.Param(
        jnp.ones(dim, param_dtype), out_sharding=shd_config.norm_weight
    )
    self.b = nnx.Param(
        jnp.zeros(dim, param_dtype), out_sharding=shd_config.norm_weight
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  @jax.named_scope("layer_norm")
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x = jnp.astype(x, jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normed = (x - mean) * jax.lax.rsqrt(var + self.norm_eps)
    out = normed * jnp.astype(self.w.value, jnp.float32) + jnp.astype(
        self.b.value, jnp.float32
    )
    return jnp.astype(out, self.dtype)


class Attention(nnx.Module):
  """OPT multi-head attention: separate biased q/k/v, scaled scores."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim
    self.scale = config.head_dim**-0.5
    hidden = config.embed_dim
    proj = config.num_heads * config.head_dim
    self.q_proj = nnx.Linear(
        hidden,
        proj,
        use_bias=True,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.column_weight
        ),
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            P(config.shd_config.column_weight[-1]),
        ),
    )
    self.k_proj = nnx.Linear(
        hidden,
        proj,
        use_bias=True,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.column_weight
        ),
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            P(config.shd_config.column_weight[-1]),
        ),
    )
    self.v_proj = nnx.Linear(
        hidden,
        proj,
        use_bias=True,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.column_weight
        ),
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            P(config.shd_config.column_weight[-1]),
        ),
    )
    self.out_proj = nnx.Linear(
        proj,
        hidden,
        use_bias=True,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.row_weight
        ),
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), P(config.shd_config.row_weight[-1])
        ),
    )

  def block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Applies the Attention computation without recomputation wrapping."""
    b, t, _ = x.shape
    seq_len = t
    query = self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim)
    key = self.k_proj(x).reshape(b, t, self.num_heads, self.head_dim)
    value = self.v_proj(x).reshape(b, t, self.num_heads, self.head_dim)

    if cache is not None:
      end_index = cache["end_index"][0]
      slice_indices = (0, end_index % cache["v"].shape[1], 0, 0)
      value = jax.lax.dynamic_update_slice(cache["v"], value, slice_indices)
      key = jax.lax.dynamic_update_slice(cache["k"], key, slice_indices)

    attn = jnp.einsum("BTND,BSND->BNTS", query, key) * self.scale
    if attn_mask is not None:
      attn = jnp.where(attn_mask[:, None, :, :], attn, K_MASK)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(value.dtype)
    out = jnp.einsum("BNTS,BSND->BTND", attn, value)
    out = out.reshape(b, t, self.num_heads * self.head_dim)
    outputs = self.out_proj(out)

    if cache is not None:
      new_cache = {
          **cache,
          "v": value,
          "k": key,
          "end_index": cache["end_index"] + seq_len,
      }
    else:
      new_cache = None
    return new_cache, outputs

  @jax.named_scope("attention")
  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Applies the block with the configured recomputation policy."""
    if self.config.remat_config in (RematConfig.BLOCK, RematConfig.BLOCK.value):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(state, x, cache, attn_mask)
    return self.block(x, cache, attn_mask)


class MLP(nnx.Module):
  """OPT dense MLP (fc1 -> act -> fc2)."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.activation = _get_activation(config.hidden_act)
    self.fc1 = nnx.Linear(
        config.embed_dim,
        config.hidden_dim,
        use_bias=True,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.column_weight
        ),
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            P(config.shd_config.column_weight[-1]),
        ),
    )
    self.fc2 = nnx.Linear(
        config.hidden_dim,
        config.embed_dim,
        use_bias=True,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.row_weight
        ),
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), P(config.shd_config.row_weight[-1])
        ),
    )

  def block(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    """Applies the MLP computation without recomputation wrapping."""
    return self.fc2(self.activation(self.fc1(x)))

  @jax.named_scope("feed_forward")
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    """Applies the block with the configured recomputation policy."""
    if self.config.remat_config in (RematConfig.BLOCK, RematConfig.BLOCK.value):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(state, x)
    return self.block(x)


class DecoderLayer(nnx.Module):
  """OPT decoder layer; residual layout depends on ``do_layer_norm_before``."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.do_layer_norm_before = config.do_layer_norm_before
    self.self_attn_layer_norm = LayerNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        shd_config=config.shd_config,
    )
    self.attn = Attention(config, rngs=rngs)
    self.final_layer_norm = LayerNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        shd_config=config.shd_config,
    )
    self.mlp = MLP(config, rngs=rngs)

  def block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Applies the DecoderLayer computation without recomputation wrapping."""
    residual = x
    attn_in = self.self_attn_layer_norm(x) if self.do_layer_norm_before else x
    cache, attn_out = self.attn(attn_in, cache, attn_mask)
    x = residual + attn_out
    if not self.do_layer_norm_before:
      x = self.self_attn_layer_norm(x)

    residual = x
    mlp_in = self.final_layer_norm(x) if self.do_layer_norm_before else x
    x = residual + self.mlp(mlp_in)
    if not self.do_layer_norm_before:
      x = self.final_layer_norm(x)
    return cache, x

  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Applies the block with the configured recomputation policy."""
    if self.config.remat_config in (
        RematConfig.DECODER,
        RematConfig.DECODER.value,
    ):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(state, x, cache, attn_mask)
    return self.block(x, cache, attn_mask)


class Embedder(nnx.Module):
  """Token + position embeddings, optional in/out projection, tied LM head."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.embed_tokens = nnx.Param(
        rngs.params.normal(
            (config.vocab_size, config.word_embed_proj_dim),
            dtype=config.param_dtype,
        ),
        out_sharding=config.shd_config.emb_vd,
    )
    self.embed_positions = nnx.Param(
        rngs.params.normal(
            (
                config.max_position_embeddings + _OPT_POS_OFFSET,
                config.embed_dim,
            ),
            dtype=config.param_dtype,
        ),
        out_sharding=config.shd_config.emb_vd,
    )
    if config.has_projection:
      self.project_in = nnx.Linear(
          config.word_embed_proj_dim,
          config.embed_dim,
          use_bias=False,
          rngs=rngs,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(), config.shd_config.column_weight
          ),
      )
      self.project_out = nnx.Linear(
          config.embed_dim,
          config.word_embed_proj_dim,
          use_bias=False,
          rngs=rngs,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(), config.shd_config.row_weight
          ),
      )
    else:
      self.project_in = None
      self.project_out = None
    self.dtype = config.dtype

  @jax.named_scope("embedder_encode")
  def encode(
      self, tokens: jaxtyping.ArrayLike, positions: jaxtyping.ArrayLike
  ) -> jaxtyping.Array:
    """Embeds tokens and positions; invalid positions produce NaNs under JIT."""
    x = self.embed_tokens[(tokens,)]
    if self.project_in is not None:
      x = self.project_in(x)
    positions = jnp.asarray(positions)
    # OPT uses position -1 for padding, which maps to embedding row 1.
    pos = jnp.where(
        positions >= -1,
        positions + _OPT_POS_OFFSET,
        self.embed_positions.shape[0],
    )
    x = x + jnp.take(self.embed_positions[...], pos, axis=0, mode="fill")
    return jnp.astype(x, self.dtype)

  @jax.named_scope("embedder_decode")
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    """Projects hidden states to vocabulary logits using tied embeddings."""
    if self.project_out is not None:
      x = self.project_out(x)
    return jnp.dot(x, self.embed_tokens.value.T)


class OPT(nnx.Module):
  """OPT decoder-only language model."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.num_embed = config.vocab_size
    self.max_position_embeddings = config.max_position_embeddings
    self.embedder = Embedder(config, rngs=rngs)
    self.layers = compat.ModuleList(
        [DecoderLayer(config, rngs=rngs) for _ in range(config.num_layers)]
    )
    if config.do_layer_norm_before:
      self.final_norm = LayerNorm(
          config.embed_dim,
          norm_eps=config.norm_eps,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
          shd_config=config.shd_config,
      )
    else:
      self.final_norm = None
    self.lm_head = (
        None
        if config.use_tied_embedding
        else nnx.Linear(
            config.word_embed_proj_dim,
            config.vocab_size,
            use_bias=False,
            rngs=rngs,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), config.shd_config.emb_dv
            ),
        )
    )

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array | None = None,  # [B, L, L']
      output_hidden_states: bool = False,
      segment_ids: jaxtyping.Array | None = None,
      skip_lm_head: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Runs the OPT decoder; returns (logits, new_cache)."""
    new_cache = None if cache is None else {}
    attention_mask, key_segments, _ = _attention_mask(
        input_tokens, cache, attention_mask, segment_ids
    )
    x = self.embedder.encode(input_tokens, positions)

    for i, layer in enumerate(self.layers):
      layer_name = f"layer_{i}"
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(x, layer_cache, attention_mask)
      if cache is not None:
        if key_segments is not None:
          layer_cache["segment_ids"] = key_segments
        new_cache[layer_name] = layer_cache
      if output_hidden_states:
        self.sow(nnx.Intermediate, "all_hidden_states", x)

    if self.final_norm is not None:
      x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, "all_hidden_states", x)
      self.sow(nnx.Intermediate, "final_hidden_state", x)
    if skip_lm_head:
      return x, new_cache

    return self.compute_final_logits(x), new_cache

  def compute_final_logits(self, x: jaxtyping.Array) -> jaxtyping.Array:
    """Projects final hidden states to float32 vocabulary logits."""
    if self.lm_head is None:
      logits = self.embedder.decode(x)
    else:
      if self.embedder.project_out is not None:
        x = self.embedder.project_out(x)
      logits = self.lm_head(x)
    return jnp.astype(logits, jnp.float32)

  def init_cache(self, batch_size: int, cache_size: int, dtype=None) -> Cache:
    """Returns empty per-layer KV caches for autoregressive generation."""
    config = self.config
    shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
    return {
        f"layer_{i}": {
            "k": jnp.zeros(shape, dtype or self.config.dtype),
            "v": jnp.zeros(shape, dtype or self.config.dtype),
            "end_index": jnp.zeros((batch_size,), jnp.int32),
            "segment_ids": jnp.zeros((batch_size, cache_size), jnp.int32),
        }
        for i in range(config.num_layers)
    }

  def get_model_input(self):
    """Dummy model input (batch size 2 for FSDP-friendly axes)."""
    dummy_batch_size, dummy_seq_len = 2, 1
    return {
        "input_tokens": jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        "positions": jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        "cache": None,
        "attention_mask": jnp.ones(
            (dummy_batch_size, dummy_seq_len, dummy_seq_len), dtype=jnp.bool
        ),
    }
