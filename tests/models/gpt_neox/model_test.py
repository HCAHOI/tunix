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

"""Numerical regression tests for GPT-NeoX attention."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.gpt_neox import model as model_lib


class AttentionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.default_matmul_precision('highest'))
    self.config = dataclasses.replace(
        model_lib.ModelConfig.pythia_14m(),
        num_layers=1,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
    )
    self.attention = model_lib.Attention(self.config, rngs=nnx.Rngs(7))
    self.inputs = jax.random.normal(jax.random.key(3), (1, 8, 16))
    self.sin, self.cos = model_lib._generate_pos_embeddings(
        jnp.arange(8)[None, :],
        self.config.rotary_ndims,
        self.config.rope_theta,
    )
    self.mask = jnp.tril(jnp.ones((1, 8, 8), dtype=jnp.bool_))
    bias = jax.random.normal(jax.random.key(5), (2, 3, 8)) * 0.1
    # Only non-rotary key biases introduce a constant per attention row.
    bias = bias.at[:, 1, self.config.rotary_ndims :].set(1e6)
    self.attention.query_key_value.bias[...] = bias.reshape(-1)

  def _forward(self, attention, inputs):
    return attention(inputs, None, self.mask, self.sin, self.cos)[1]

  def _float64_reference(self):
    """Evaluates the original biased attention formula in NumPy float64."""
    projection = self.attention.query_key_value
    qkv = np.asarray(self.inputs, np.float64) @ np.asarray(projection.kernel)
    qkv += np.asarray(projection.bias)
    q, k, v = np.split(qkv.reshape(1, 8, 2, 24), 3, axis=-1)
    n = self.config.rotary_ndims
    sin, cos = (
        np.asarray(self.sin)[:, :, None],
        np.asarray(self.cos)[:, :, None],
    )

    def rotate(x):
      a, b = x[..., : n // 2], x[..., n // 2 : n]
      return np.concatenate(
          (a * cos - b * sin, b * cos + a * sin, x[..., n:]), axis=-1
      )

    scores = np.einsum('btnd,bsnd->bnts', rotate(q), rotate(k)) / np.sqrt(8)
    scores = np.where(np.asarray(self.mask)[:, None], scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    output = np.einsum('bnts,bsnd->btnd', weights, v).reshape(1, 8, 16)
    return output @ np.asarray(self.attention.dense.kernel) + np.asarray(
        self.attention.dense.bias
    )

  @parameterized.parameters(False, True)
  def test_large_key_bias_matches_float64_reference(self, use_jit):
    forward = nnx.jit(self._forward) if use_jit else self._forward
    actual = forward(self.attention, self.inputs)
    np.testing.assert_allclose(
        actual, self._float64_reference(), atol=1e-5, rtol=1e-5
    )
    bias = self.attention.query_key_value.bias[...].reshape(2, 3, 8)
    self.attention.query_key_value.bias[...] = (
        bias.at[:, 1, self.config.rotary_ndims :].set(0).reshape(-1)
    )
    np.testing.assert_array_equal(actual, forward(self.attention, self.inputs))

  def test_key_bias_does_not_change_input_gradients(self):
    def loss(attention, inputs):
      return jnp.sum(self._forward(attention, inputs) ** 2)

    gradient = nnx.jit(nnx.grad(loss, argnums=1))
    actual = gradient(self.attention, self.inputs)
    bias = self.attention.query_key_value.bias[...].reshape(2, 3, 8)
    self.attention.query_key_value.bias[...] = (
        bias.at[:, 1, self.config.rotary_ndims :].set(0).reshape(-1)
    )
    expected = gradient(self.attention, self.inputs)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
  absltest.main()
