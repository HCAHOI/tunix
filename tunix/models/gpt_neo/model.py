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

"""GPT-Neo decoder-only language model."""

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
  """Parameter partition specs for the GPT-Neo model."""

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
  """Configuration for the GPT-Neo model."""

  num_layers: int
  vocab_size: int
  embed_dim: int  # hidden_size
  hidden_dim: int  # intermediate_size (== 4 * embed_dim for GPT-Neo)
  num_heads: int
  head_dim: int
  max_position_embeddings: int
  window_size: int
  # Per-layer attention pattern; each entry is "global" or "local".
  attention_layers: tuple[str, ...]
  norm_eps: float  # layer_norm_epsilon
  hidden_act: str = "gelu_new"
  use_tied_embedding: bool = True
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE

  @property
  def num_kv_heads(self) -> int:  # GPT-Neo is full MHA.
    """Number of key/value heads for multi-head attention."""
    return self.num_heads

  @classmethod
  def gpt_neo_125m(cls):
    """Returns GPT-Neo-125M with alternating global and local attention."""
    return cls(
        num_layers=12,
        vocab_size=50257,
        embed_dim=768,
        hidden_dim=3072,
        num_heads=12,
        head_dim=64,
        max_position_embeddings=2048,
        window_size=256,
        attention_layers=("global", "local") * 6,
        norm_eps=1e-05,
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
  """GPT-Neo multi-head attention: separate q/k/v, unscaled scores."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim
    hidden = config.embed_dim
    proj = config.num_heads * config.head_dim
    # GPT-Neo: q/k/v projections have NO bias; out_proj HAS bias.
    self.q_proj = nnx.Linear(
        hidden,
        proj,
        use_bias=False,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.column_weight
        ),
    )
    self.k_proj = nnx.Linear(
        hidden,
        proj,
        use_bias=False,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.column_weight
        ),
    )
    self.v_proj = nnx.Linear(
        hidden,
        proj,
        use_bias=False,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), config.shd_config.column_weight
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

    # GPT-Neo does NOT scale the attention scores; HF upcasts q/k to float32.
    attn = jnp.einsum(
        "BTND,BSND->BNTS",
        query.astype(jnp.float32),
        key.astype(jnp.float32),
    )
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
  """GPT-Neo dense gelu_new MLP."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.activation = _get_activation(config.hidden_act)
    self.c_fc = nnx.Linear(
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
    self.c_proj = nnx.Linear(
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
    return self.c_proj(self.activation(self.c_fc(x)))

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
  """GPT-Neo decoder layer (pre-LN, sequential residual)."""

  def __init__(
      self, config: ModelConfig, attention_type: str, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.attention_type = attention_type
    self.ln_1 = LayerNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        shd_config=config.shd_config,
    )
    self.attn = Attention(config, rngs=rngs)
    self.ln_2 = LayerNorm(
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
    cache, attn_out = self.attn(self.ln_1(x), cache, attn_mask)
    x = x + attn_out
    outputs = x + self.mlp(self.ln_2(x))
    return cache, outputs

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
  """Learned token + absolute position embeddings; tied LM head."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      max_position_embeddings: int,
      *,
      rngs: nnx.Rngs,
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.input_embedding = nnx.Param(
        rngs.params.normal((vocab_size, embed_dim), dtype=param_dtype),
        out_sharding=shd_config.emb_vd,
    )
    self.position_embedding = nnx.Param(
        rngs.params.normal(
            (max_position_embeddings, embed_dim), dtype=param_dtype
        ),
        out_sharding=shd_config.emb_vd,
    )
    self.dtype = dtype

  @jax.named_scope("embedder_encode")
  def encode(
      self, tokens: jaxtyping.ArrayLike, positions: jaxtyping.ArrayLike
  ) -> jaxtyping.Array:
    """Embeds tokens and positions; invalid positions produce NaNs under JIT."""
    positions = jnp.asarray(positions)
    # Disable negative-index wrapping and silent clipping of positive overflow.
    positions = jnp.where(
        positions >= 0, positions, self.position_embedding.shape[0]
    )
    position_embeddings = jnp.take(
        self.position_embedding[...], positions, axis=0, mode="fill"
    )
    x = self.input_embedding[(tokens,)] + position_embeddings
    return jnp.astype(x, self.dtype)

  @jax.named_scope("embedder_decode")
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    """Projects hidden states to vocabulary logits using tied embeddings."""
    return jnp.dot(x, self.input_embedding.value.T)


class GPTNeo(nnx.Module):
  """GPT-Neo decoder-only language model."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.num_embed = config.vocab_size
    self.max_position_embeddings = config.max_position_embeddings
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        max_position_embeddings=config.max_position_embeddings,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        shd_config=config.shd_config,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(config, config.attention_layers[i], rngs=rngs)
        for i in range(config.num_layers)
    ])
    self.final_norm = LayerNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        shd_config=config.shd_config,
    )
    self.lm_head = (
        None
        if config.use_tied_embedding
        else nnx.Linear(
            config.embed_dim,
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

  def _local_mask(
      self, attn_mask: jaxtyping.Array, query_indices: jaxtyping.Array
  ) -> jaxtyping.Array:
    """AND the base mask with a local (windowed) band for local layers.

    Position i attends to key j only if ``i - window_size < j <= i`` (in
    addition to the causal/padding constraints already in ``attn_mask``).
    """
    q_idx = query_indices[:, None]
    k_idx = jnp.arange(attn_mask.shape[-1])[None, :]
    band = k_idx > (q_idx - self.config.window_size)
    # ``attn_mask`` may be float (0/1) from make_causal_attn_mask; use
    # bool for &.
    return attn_mask.astype(jnp.bool_) & band[None, :, :]

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
    """Runs the GPT-Neo decoder; returns (logits, new_cache)."""
    new_cache = None if cache is None else {}
    x = self.embedder.encode(input_tokens, positions)

    attention_mask, key_segments, query_indices = _attention_mask(
        input_tokens, cache, attention_mask, segment_ids
    )
    local_mask = self._local_mask(attention_mask, query_indices)

    for i, layer in enumerate(self.layers):
      layer_name = f"layer_{i}"
      layer_cache = cache[layer_name] if cache else None
      mask = local_mask if layer.attention_type == "local" else attention_mask
      layer_cache, x = layer(x, layer_cache, mask)
      if cache is not None:
        if key_segments is not None:
          layer_cache["segment_ids"] = key_segments
        new_cache[layer_name] = layer_cache
      if output_hidden_states:
        self.sow(nnx.Intermediate, "all_hidden_states", x)

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, "all_hidden_states", x)
      self.sow(nnx.Intermediate, "final_hidden_state", x)
    if skip_lm_head:
      return x, new_cache

    return self.compute_final_logits(x), new_cache

  def compute_final_logits(self, x: jaxtyping.Array) -> jaxtyping.Array:
    """Projects final hidden states to float32 vocabulary logits."""
    logits = (
        self.embedder.decode(x) if self.lm_head is None else self.lm_head(x)
    )
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
