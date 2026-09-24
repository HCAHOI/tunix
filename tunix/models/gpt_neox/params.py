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

"""Utils for loading and converting GPT-NeoX / Pythia PT weights."""

import pathlib

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.gpt_neox import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule))."""
  # PyTorch nn.Linear weights are [out, in]; nnx.Linear kernels are [in, out],
  # so linear weights are transposed with permute (1, 0). The fused
  # query_key_value weight is kept fused (transposed only); the per-head
  # q/k/v split happens at forward time, matching HF's
  # view(..., num_heads, 3*head_dim).
  mapping = {
      r"gpt_neox\.embed_in\.weight$": ("embedder.input_embedding", None),
      r"embed_out\.weight$": ("lm_head.kernel", ((1, 0), None)),
      r"gpt_neox\.final_layer_norm\.weight$": ("final_norm.w", None),
      r"gpt_neox\.final_layer_norm\.bias$": ("final_norm.b", None),
      r"gpt_neox\.layers\.([0-9]+)\.input_layernorm\.weight$": (
          r"layers.\1.input_layernorm.w",
          None,
      ),
      r"gpt_neox\.layers\.([0-9]+)\.input_layernorm\.bias$": (
          r"layers.\1.input_layernorm.b",
          None,
      ),
      r"gpt_neox\.layers\.([0-9]+)\.post_attention_layernorm\.weight$": (
          r"layers.\1.post_attention_layernorm.w",
          None,
      ),
      r"gpt_neox\.layers\.([0-9]+)\.post_attention_layernorm\.bias$": (
          r"layers.\1.post_attention_layernorm.b",
          None,
      ),
      r"gpt_neox\.layers\.([0-9]+)\.attention\.query_key_value\.weight$": (
          r"layers.\1.attn.query_key_value.kernel",
          ((1, 0), None),
      ),
      r"gpt_neox\.layers\.([0-9]+)\.attention\.query_key_value\.bias$": (
          r"layers.\1.attn.query_key_value.bias",
          None,
      ),
      r"gpt_neox\.layers\.([0-9]+)\.attention\.dense\.weight$": (
          r"layers.\1.attn.dense.kernel",
          ((1, 0), None),
      ),
      r"gpt_neox\.layers\.([0-9]+)\.attention\.dense\.bias$": (
          r"layers.\1.attn.dense.bias",
          None,
      ),
      r"gpt_neox\.layers\.([0-9]+)\.mlp\.dense_h_to_4h\.weight$": (
          r"layers.\1.mlp.dense_h_to_4h.kernel",
          ((1, 0), None),
      ),
      r"gpt_neox\.layers\.([0-9]+)\.mlp\.dense_h_to_4h\.bias$": (
          r"layers.\1.mlp.dense_h_to_4h.bias",
          None,
      ),
      r"gpt_neox\.layers\.([0-9]+)\.mlp\.dense_4h_to_h\.weight$": (
          r"layers.\1.mlp.dense_4h_to_h.kernel",
          ((1, 0), None),
      ),
      r"gpt_neox\.layers\.([0-9]+)\.mlp\.dense_4h_to_h\.bias$": (
          r"layers.\1.mlp.dense_4h_to_h.bias",
          None,
      ),
  }
  if cfg.use_tied_embedding:
    del mapping[r"embed_out\.weight$"]
  return mapping


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
) -> model_lib.GPTNeoX:
  """Load tensors from the safetensors file and create a GPT-NeoX model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.GPTNeoX,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=None,
      dtype=dtype,
      mode=mode,
  )


def _state_key_to_safetensors_key(lora_name: str) -> str:
  """Maps a LoRA module path to its Hugging Face weight name."""
  if lora_name == "lm_head":
    return "embed_out.weight"
  return f"gpt_neox.{lora_name}.weight".replace(".attn.", ".attention.")


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.GPTNeoX,
    rank: int,
    alpha: float,
) -> None:
  """Saves merged LoRA weights using Tunix's single-file safetensors saver.

  Args:
    local_model_path: Directory containing one model.safetensors checkpoint.
    output_dir: Destination directory for merged weights and metadata.
    lora_model: Model with LoRA adapters.
    rank: LoRA rank.
    alpha: LoRA scaling factor.

  Raises:
    ValueError: The output would overwrite the source, or the checkpoint does
      not contain a single model.safetensors file.
  """
  source = pathlib.Path(local_model_path).resolve()
  output = pathlib.Path(output_dir).resolve()
  if source == output or output in source.parents:
    raise ValueError("Merged output must not overwrite the source checkpoint")
  if not (source / "model.safetensors").is_file():
    raise ValueError("LoRA merge requires a single model.safetensors file")
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=_state_key_to_safetensors_key,
      transpose_rules={"weight": (1, 0)},
  )
