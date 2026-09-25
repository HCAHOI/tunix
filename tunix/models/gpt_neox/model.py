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

"""GPT-NeoX / Pythia decoder-only language model."""

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
  """Parameter partition specs for the GPT-NeoX model."""

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
  """Configuration for the GPT-NeoX / Pythia model."""

  num_layers: int
  vocab_size: int
  embed_dim: int  # hidden_size
  hidden_dim: int  # intermediate_size
  num_heads: int
  head_dim: int
  rotary_pct: float
  rope_theta: int
  norm_eps: float  # layer_norm_eps
  use_parallel_residual: bool = True
  use_tied_embedding: bool = False  # GPT-NeoX has a separate embed_out head
  hidden_act: str = "gelu"
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE

  @property
  def rotary_ndims(self) -> int:
    """Number of head dimensions to which rotary embeddings apply."""
    return int(self.head_dim * self.rotary_pct)

  @property
  def num_kv_heads(self) -> int:  # GPT-NeoX is full MHA
    """Number of key/value heads for multi-head attention."""
    return self.num_heads

  @classmethod
  def pythia_14m(cls):
    """Returns the registered pythia_14m configuration."""
    return cls(
        num_layers=6,
        vocab_size=50304,
        embed_dim=128,
        hidden_dim=512,
        num_heads=4,
        head_dim=32,
        rotary_pct=0.25,
        rope_theta=10000,
        norm_eps=1e-05,
    )

  @classmethod
  def pythia_70m(cls):
    """Returns the registered pythia_70m configuration."""
    return cls(
        num_layers=6,
        vocab_size=50304,
        embed_dim=512,
        hidden_dim=2048,
        num_heads=8,
        head_dim=64,
        rotary_pct=0.25,
        rope_theta=10000,
        norm_eps=1e-05,
    )

  @classmethod
  def pythia_160m(cls):
    """Returns the registered pythia_160m configuration."""
    return cls(
        num_layers=12,
        vocab_size=50304,
        embed_dim=768,
        hidden_dim=3072,
        num_heads=12,
        head_dim=64,
        rotary_pct=0.25,
        rope_theta=10000,
        norm_eps=1e-05,
    )


def _generate_pos_embeddings(
    positions: jax.Array, features: int, rope_theta: int
) -> tuple[jax.Array, jax.Array]:
  """Sin/Cos for rotary embeddings over ``features`` dims (half returned).

  ``features`` is ``rotary_ndims`` for GPT-NeoX partial rotary. Frequencies
  use float32 and highest-precision multiplication before sin/cos.
  """
  fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
  timescale = rope_theta**fraction
  rotational_frequency = 1.0 / timescale
  sinusoid_inp = jnp.einsum(
      "BT,k->BTk",
      positions,
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST,
  )
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rotary_embedding(
    x: jax.Array, sin: jax.Array, cos: jax.Array
) -> jax.Array:
  """rotate_half RoPE (GPT-NeoX/Llama convention).

  x: [B,T,H,D], sin/cos: [B,T,D/2].
  """
  assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  sin, cos = sin[:, :, None, :], cos[:, :, None, :]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


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


class QKVProjection(nnx.Linear):
  """Fused QKV projection without the softmax-invariant key bias.

  Non-rotary key biases add the same constant to every unmasked score in an
  attention row. Large pretrained biases can erase score differences in
  float32. Omit that part before adding the bias, preserving query, value,
  and rotary key biases. The full parameter is retained for checkpoint
  compatibility; its unused components have zero gradients.
  """

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    super().__init__(
        config.embed_dim,
        3 * config.num_heads * config.head_dim,
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
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim
    self.rotary_ndims = config.rotary_ndims

  def __call__(self, inputs: jax.Array) -> jax.Array:
    bias = self.bias[...].reshape(self.num_heads, 3, self.head_dim)
    bias = bias.at[:, 1, self.rotary_ndims :].set(0).reshape(-1)
    inputs, kernel, bias = self.promote_dtype(
        (inputs, self.kernel[...], bias), dtype=self.dtype
    )
    output = self.dot_general(
        inputs,
        kernel,
        (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=self.precision,
    )
    return output + bias


class Attention(nnx.Module):
  """GPT-NeoX multi-head attention with fused QKV and partial rotary."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim
    self.rotary_ndims = config.rotary_ndims
    self.scale = config.head_dim**-0.5
    hidden = config.embed_dim
    self.query_key_value = QKVProjection(config, rngs=rngs)
    self.dense = nnx.Linear(
        config.num_heads * config.head_dim,
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

  def _rope(self, x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    x_rot, x_pass = x[..., : self.rotary_ndims], x[..., self.rotary_ndims :]
    x_rot = apply_rotary_embedding(x_rot, sin, cos)
    return jnp.concatenate([x_rot, x_pass], axis=-1)

  def block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Applies the Attention computation without recomputation wrapping."""
    b, t, _ = x.shape
    seq_len = t
    qkv = self.query_key_value(x)  # [B, T, 3*num_heads*head_dim]
    qkv = qkv.reshape(b, t, self.num_heads, 3 * self.head_dim)
    query, key, value = jnp.split(qkv, 3, axis=-1)  # each [B,T,N,head_dim]

    query = self._rope(query, sin, cos)
    key = self._rope(key, sin, cos)

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
    outputs = self.dense(out)

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
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Applies the block with the configured recomputation policy."""
    if self.config.remat_config in (RematConfig.BLOCK, RematConfig.BLOCK.value):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state, x, cache, attn_mask, sin, cos
      )
    return self.block(x, cache, attn_mask, sin, cos)


class MLP(nnx.Module):
  """GPT-NeoX dense GELU MLP."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.activation = _get_activation(config.hidden_act)
    self.dense_h_to_4h = nnx.Linear(
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
    self.dense_4h_to_h = nnx.Linear(
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
    return self.dense_4h_to_h(self.activation(self.dense_h_to_4h(x)))

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
  """GPT-NeoX decoder layer with parallel (or sequential) residual."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.input_layernorm = LayerNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        shd_config=config.shd_config,
    )
    self.post_attention_layernorm = LayerNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        shd_config=config.shd_config,
    )
    self.attn = Attention(config, rngs=rngs)
    self.mlp = MLP(config, rngs=rngs)

  def block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Applies the DecoderLayer computation without recomputation wrapping."""
    attn_in = self.input_layernorm(x)
    cache, attn_out = self.attn(attn_in, cache, attn_mask, sin, cos)
    if self.config.use_parallel_residual:
      # Pythia: attention and MLP both read the layer input.
      mlp_out = self.mlp(self.post_attention_layernorm(x))
      outputs = x + attn_out + mlp_out
    else:
      x = x + attn_out
      outputs = x + self.mlp(self.post_attention_layernorm(x))
    return cache, outputs

  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
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

      return jax.checkpoint(_checkpointed_block)(
          state, x, cache, attn_mask, sin, cos
      )
    return self.block(x, cache, attn_mask, sin, cos)


class Embedder(nnx.Module):
  """Token embedder (no scaling, unlike Gemma)."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
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
    self.dtype = dtype

  @jax.named_scope("embedder_encode")
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    """Embeds token IDs into the configured compute dtype."""
    return jnp.astype(self.input_embedding[(x,)], self.dtype)


class GPTNeoX(nnx.Module):
  """GPT-NeoX / Pythia decoder-only language model."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.num_embed = config.vocab_size
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        shd_config=config.shd_config,
    )
    self.layers = compat.ModuleList(
        [DecoderLayer(config, rngs=rngs) for _ in range(config.num_layers)]
    )
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
    """Runs the GPT-NeoX decoder; returns (logits, new_cache)."""
    new_cache = None if cache is None else {}
    attention_mask, key_segments, _ = _attention_mask(
        input_tokens, cache, attention_mask, segment_ids
    )
    x = self.embedder.encode(input_tokens)
    sin, cos = _generate_pos_embeddings(
        positions, self.config.rotary_ndims, self.config.rope_theta
    )
    sin, cos = sin.astype(x.dtype), cos.astype(x.dtype)

    for i, layer in enumerate(self.layers):
      layer_name = f"layer_{i}"
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(x, layer_cache, attention_mask, sin, cos)
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
        jnp.dot(x, self.embedder.input_embedding[...].T)
        if self.lm_head is None
        else self.lm_head(x)
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
