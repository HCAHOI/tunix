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

"""Offline checkpoint and native interface tests for GPT-NeoX, GPT-Neo, and OPT."""

import dataclasses
import pathlib
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
import safetensors.numpy as safe_np
import torch
import transformers
from tunix.generate import sampler as sampler_lib
from tunix.models.gpt_neo import model as neo_model
from tunix.models.gpt_neo import params as neo_params
from tunix.models.gpt_neox import model as neox_model
from tunix.models.gpt_neox import params as neox_params
from tunix.models.opt import model as opt_model
from tunix.models.opt import params as opt_params
from tunix.rl import utils as rl_utils
from tunix.sft import peft_trainer
from tunix.tests import test_common

_FAMILIES = {
    "gpt_neox": (neox_model, neox_model.GPTNeoX, neox_params),
    "gpt_neo": (neo_model, neo_model.GPTNeo, neo_params),
    "opt": (opt_model, opt_model.OPT, opt_params),
}
_LORA_MODULES = (
    ".*(query_key_value|dense|dense_h_to_4h|dense_4h_to_h|q_proj|k_proj|"
    "v_proj|out_proj|c_fc|c_proj|fc1|fc2|project_in|project_out|lm_head)"
)


def _config(family):
  common = dict(
      num_layers=2,
      vocab_size=32,
      embed_dim=16,
      hidden_dim=32,
      num_heads=2,
      head_dim=8,
      norm_eps=1e-5,
  )
  if family == "gpt_neox":
    return neox_model.ModelConfig(**common, rotary_pct=0.25, rope_theta=10000)
  if family == "gpt_neo":
    return neo_model.ModelConfig(
        **common,
        max_position_embeddings=16,
        window_size=4,
        attention_layers=("global", "local"),
    )
  return opt_model.ModelConfig(
      **common, max_position_embeddings=16, word_embed_proj_dim=8
  )


def _hf_config(family, config):
  common = dict(
      vocab_size=config.vocab_size,
      hidden_size=config.embed_dim,
      intermediate_size=config.hidden_dim,
      num_hidden_layers=config.num_layers,
      num_attention_heads=config.num_heads,
      max_position_embeddings=16,
      tie_word_embeddings=config.use_tied_embedding,
      pad_token_id=1,
      attn_implementation="eager",
  )
  if family == "gpt_neox":
    return transformers.GPTNeoXConfig(
        **common,
        rotary_pct=config.rotary_pct,
        layer_norm_eps=config.norm_eps,
        hidden_act=config.hidden_act,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
  if family == "gpt_neo":
    return transformers.GPTNeoConfig(
        **common,
        num_layers=2,
        num_heads=2,
        attention_types=[[["global", "local"], 1]],
        window_size=4,
        layer_norm_epsilon=config.norm_eps,
        activation_function=config.hidden_act,
        attention_dropout=0.0,
        resid_dropout=0.0,
        embed_dropout=0.0,
    )
  return transformers.OPTConfig(
      **common,
      ffn_dim=config.hidden_dim,
      word_embed_proj_dim=config.word_embed_proj_dim,
      do_layer_norm_before=config.do_layer_norm_before,
      activation_function=config.hidden_act,
      dropout=0.0,
      attention_dropout=0.0,
  )


def _adapt(model):
  return qwix.apply_lora_to_model(
      model,
      qwix.LoraProvider(module_path=_LORA_MODULES, rank=2, alpha=4.0),
      rngs=nnx.Rngs(7),
      **model.get_model_input(),
  )


class DecoderModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = jax.make_mesh(
        (1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2
    )
    self.enterContext(jax.set_mesh(self.mesh))
    self.enterContext(jax.default_matmul_precision("highest"))
    self.tokens = jnp.array([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=jnp.int32)
    self.positions = jnp.arange(8, dtype=jnp.int32)[None, :]

  def _checkpoint(
      self, family, *, legacy=False, original_prefix=False, tied=None
  ):
    config = _config(family)
    if tied is not None:
      config = dataclasses.replace(config, use_tied_embedding=tied)
    torch.manual_seed(17)
    reference = transformers.AutoModelForCausalLM.from_config(
        _hf_config(family, config)
    ).eval()
    path = pathlib.Path(self.create_tempdir().full_path)
    state = {
        k: v.detach().contiguous().clone()
        for k, v in reference.state_dict().items()
    }
    if original_prefix:
      state = {k.removeprefix("model."): v for k, v in state.items()}
    if legacy:
      torch.save(state, path / "pytorch_model.bin")
    else:
      safe_np.save_file(
          {k: v.numpy() for k, v in state.items()},
          str(path / "model.safetensors"),
      )
    reference.config.to_json_file(path / "config.json")
    return config, reference, path

  def _load(self, family, path, config, mode="optimized"):
    return _FAMILIES[family][2].create_model_from_safe_tensors(
        str(path), config, mesh=self.mesh, mode=mode
    )

  def _logits(self, model):
    return model(self.tokens, self.positions, None)[0]

  @parameterized.product(
      family=tuple(_FAMILIES),
      mode=("optimized", "original"),
      padding=("left", "right"),
  )
  def test_loader_matches_hugging_face(self, family, mode, padding):
    config, reference, path = self._checkpoint(family)
    model = self._load(family, path, config, mode)
    tokens = np.array(
        [[1, 1, 2, 3, 4, 5, 6, 7]]
        if padding == "left"
        else [[2, 3, 4, 5, 6, 7, 1, 1]],
        np.int32,
    )
    mask = tokens != 1
    positions = np.maximum(np.cumsum(mask, axis=-1) - 1, 0).astype(np.int32)
    kwargs = (
        {} if family == "opt" else {"position_ids": torch.tensor(positions)}
    )
    with torch.no_grad():
      expected = reference(
          torch.tensor(tokens, dtype=torch.long),
          attention_mask=torch.tensor(mask),
          **kwargs,
      ).logits.numpy()
    actual = nnx.jit(lambda m, t, p, a: m(t, p, None, attention_mask=a)[0])(
        model,
        jnp.asarray(tokens),
        jnp.asarray(positions),
        jnp.asarray(mask[:, None, :]),
    )
    np.testing.assert_allclose(
        np.asarray(actual)[mask], expected[mask], atol=2e-5, rtol=2e-5
    )

  def _assert_merge_roundtrip(self, family, config, source):
    params = _FAMILIES[family][2]
    model = _adapt(self._load(family, source, config))
    before = np.asarray(self._logits(model))
    adapters = jax.tree.leaves(nnx.state(model, nnx.LoRAParam))
    self.assertNotEmpty(adapters)
    rngs = nnx.Rngs(11)
    for _, variable in nnx.iter_graph(model):
      if isinstance(variable, nnx.LoRAParam):
        variable[...] = 0.05 * rngs.params.normal(variable.shape)
    expected = self._logits(model)
    self.assertGreater(float(jnp.max(jnp.abs(expected - before))), 1e-6)
    output = pathlib.Path(self.create_tempdir().full_path) / "merged"
    source_files = {p.name: p.read_bytes() for p in source.iterdir()}
    params.save_lora_merged_model_as_safetensors(
        str(source), str(output), model, 2, 4.0
    )
    actual = self._logits(self._load(family, output, config))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    self.assertEqual(
        (output / "config.json").read_bytes(), source_files["config.json"]
    )
    self.assertEqual(
        {p.name: p.read_bytes() for p in source.iterdir()}, source_files
    )
    self.assertFalse((output / "pytorch_model.bin").exists())

  @parameterized.product(family=tuple(_FAMILIES), tied=(False, True))
  def test_lora_merge_roundtrip(self, family, tied):
    config, _, path = self._checkpoint(family, tied=tied)
    self._assert_merge_roundtrip(family, config, path)

  @parameterized.parameters(False, True)
  def test_opt_bin_load_and_merge(self, original_prefix):
    config, reference, path = self._checkpoint(
        "opt", legacy=True, original_prefix=original_prefix
    )
    model = self._load("opt", path, config)
    with torch.no_grad():
      expected = reference(
          torch.tensor(np.asarray(self.tokens), dtype=torch.long)
      ).logits.numpy()
    np.testing.assert_allclose(
        self._logits(model), expected, atol=2e-5, rtol=2e-5
    )
    self._assert_merge_roundtrip("opt", config, path)

  def test_opt_bin_requires_legacy_dependency(self):
    config, _, path = self._checkpoint("opt", legacy=True)
    with mock.patch.dict(sys.modules, {"torch": None}):
      with self.assertRaisesRegex(ImportError, r"google-tunix\[legacy\]"):
        self._load("opt", path, config)

  def test_opt_conversion_cleans_temporary_file(self):
    _, _, path = self._checkpoint("opt", legacy=True)
    with self.assertRaisesRegex(RuntimeError, "consumer failed"):
      with opt_params._safetensors_directory(str(path)) as converted:
        self.assertTrue(
            (pathlib.Path(converted) / "model.safetensors").is_file()
        )
        raise RuntimeError("consumer failed")
    self.assertFalse(pathlib.Path(converted).exists())
    self.assertTrue((path / "pytorch_model.bin").is_file())

  @parameterized.product(
      family=tuple(_FAMILIES), destination=("source", "parent", "sharded")
  )
  def test_unsafe_export_preserves_existing_files(self, family, destination):
    config, _, source = self._checkpoint(family)
    model = self._load(family, source, config)
    if destination == "sharded":
      (source / "model.safetensors").rename(
          source / "model-00001-of-00002.safetensors"
      )
      output = pathlib.Path(self.create_tempdir().full_path)
    else:
      output = source if destination == "source" else source.parent
    sentinel = output / "keep.txt"
    sentinel.write_text("preserved")
    with self.assertRaises(ValueError):
      _FAMILIES[family][2].save_lora_merged_model_as_safetensors(
          str(source), str(output), model, 2, 4.0
      )
    self.assertEqual(sentinel.read_text(), "preserved")

  def _model(self, family):
    return _FAMILIES[family][1](_config(family), rngs=nnx.Rngs(3))

  def _sampler(self, family):
    vocabulary = test_common.MockVocab()
    config = dataclasses.replace(
        _config(family), vocab_size=vocabulary.GetPieceSize()
    )
    model = _FAMILIES[family][1](config, rngs=nnx.Rngs(3))
    return sampler_lib.Sampler(
        model,
        vocabulary,
        sampler_lib.CacheConfig(
            cache_size=32,
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
        ),
    )

  @parameterized.parameters(*_FAMILIES)
  def test_cached_decode_matches_full_sequence(self, family):
    model = self._model(family)
    expected = self._logits(model)
    cache = model.init_cache(1, 32, jnp.float32)
    forward = nnx.jit(lambda m, t, p, c: m(t, p, c))
    _, cache = forward(model, self.tokens[:, :4], self.positions[:, :4], cache)
    actual, _ = forward(model, self.tokens[:, 4:], self.positions[:, 4:], cache)
    np.testing.assert_allclose(actual, expected[:, 4:], atol=2e-5, rtol=2e-5)

  @parameterized.parameters(*_FAMILIES)
  def test_native_sampler_batch_matches_individual_prompts(self, family):
    sampler = self._sampler(family)
    prompts = [[3, 4, 5], [6, 7]]
    batched = sampler(
        prompt_token_ids=prompts, max_generation_steps=3, return_logits=True
    )
    for index, prompt in enumerate(prompts):
      single = sampler(
          prompt_token_ids=[prompt], max_generation_steps=3, return_logits=True
      )
      np.testing.assert_array_equal(batched.tokens[index], single.tokens[0])
      self.assertTrue(np.all(np.isfinite(batched.logits[index])))

  @parameterized.product(family=tuple(_FAMILIES), use_lora=(False, True))
  def test_native_training_updates_parameters(self, family, use_lora):
    model = self._model(family)
    if use_lora:
      model = _adapt(model)
    base_filter = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))
    base_before = [
        np.array(x) for x in jax.tree.leaves(nnx.state(model, base_filter))
    ]
    trainable_filter = nnx.LoRAParam if use_lora else nnx.Param
    before = [
        np.array(x) for x in jax.tree.leaves(nnx.state(model, trainable_filter))
    ]
    config = peft_trainer.TrainingConfig(max_steps=2, eval_every_n_steps=10)
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    batch = dict(
        input_tokens=self.tokens,
        input_mask=jnp.ones_like(self.tokens),
        positions=self.positions,
        attention_mask=jnp.tril(jnp.ones((1, 8, 8), dtype=jnp.bool_)),
    )
    trainer.train([batch, batch])
    self.assertEqual(trainer.train_steps, 2)
    after = jax.tree.leaves(nnx.state(trainer.model, trainable_filter))
    self.assertTrue(all(np.all(np.isfinite(x)) for x in after))
    self.assertTrue(
        any(np.any(a != b) for a, b in zip(before, after, strict=True))
    )
    if use_lora:
      base_after = jax.tree.leaves(nnx.state(trainer.model, base_filter))
      for a, b in zip(base_before, base_after, strict=True):
        np.testing.assert_array_equal(a, b)

  @parameterized.parameters(*_FAMILIES)
  def test_native_score_head_uses_sharding_config(self, family):
    model = rl_utils.TransformerWithScoreHead(
        self._model(family), rngs=nnx.Rngs(5)
    )
    scores = model(self.tokens, self.positions, None)
    self.assertEqual(scores.shape, (1, 8, 1))
    self.assertTrue(np.all(np.isfinite(scores)))

  @parameterized.product(family=tuple(_FAMILIES), level=("BLOCK", "DECODER"))
  def test_remat_preserves_gradients(self, family, level):
    model = self._model(family)
    loss = nnx.jit(nnx.value_and_grad(lambda m: jnp.mean(self._logits(m) ** 2)))
    expected, expected_grad = loss(model)
    model.config.remat_config = getattr(_FAMILIES[family][0].RematConfig, level)
    actual, actual_grad = loss(model)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    for a, b in zip(
        jax.tree.leaves(actual_grad),
        jax.tree.leaves(expected_grad),
        strict=True,
    ):
      np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-4)

  @parameterized.product(family=("gpt_neo", "opt"), use_jit=(False, True))
  def test_invalid_positions_do_not_clip_or_wrap(self, family, use_jit):
    model = self._model(family)
    positions = jnp.array([[-2, -1, 0, 15, 16, 2048, 3000]], dtype=jnp.int32)
    encode = lambda m, p: m.embedder.encode(jnp.zeros_like(p), p)
    if use_jit:
      encode = nnx.jit(encode)
    output = encode(model, positions)
    valid = np.array(
        [[False, family == "opt", True, True, False, False, False]]
    )
    self.assertTrue(np.all(np.isfinite(np.asarray(output)[valid])))
    self.assertTrue(np.all(np.isnan(np.asarray(output)[~valid])))

  @parameterized.product(family=("gpt_neo", "opt"), prompt_length=(2, 17))
  def test_sampler_rejects_position_overflow(self, family, prompt_length):
    sampler = self._sampler(family)
    steps = 16 if prompt_length == 2 else 1
    with self.assertRaisesRegex(ValueError, "model position limit of 16"):
      sampler(
          prompt_token_ids=[[3] * prompt_length], max_generation_steps=steps
      )

  @parameterized.parameters("gpt_neo", "opt")
  def test_sampler_allows_final_position_and_unused_cache(self, family):
    sampler = self._sampler(family)
    result = sampler(
        prompt_token_ids=[[3] * 16], max_generation_steps=1, return_logits=True
    )
    self.assertLen(result.tokens[0], 1)
    self.assertTrue(np.all(np.isfinite(result.logits[0])))
    # Left-padding slots exceed the table size, but their positions do not.
    result = sampler(
        ["hello"],
        max_prompt_length=20,
        max_generation_steps=2,
        return_logits=True,
    )
    self.assertTrue(np.all(np.isfinite(result.logits[0])))


if __name__ == "__main__":
  absltest.main()
