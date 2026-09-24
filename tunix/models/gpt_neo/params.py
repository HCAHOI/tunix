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

"""Utils for loading and converting GPT-Neo PT weights."""

import pathlib

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.gpt_neo import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Returns HF-to-NNX parameter names and transforms for GPT-Neo.

  Linear weights transpose from [out, in] to [in, out]. Embeddings load as-is;
  checkpoint causal-mask buffers are skipped by the loader.
  """
  mapping = {
      r"transformer\.wte\.weight$": ("embedder.input_embedding", None),
      r"transformer\.wpe\.weight$": ("embedder.position_embedding", None),
      r"transformer\.ln_f\.weight$": ("final_norm.w", None),
      r"transformer\.ln_f\.bias$": ("final_norm.b", None),
      r"transformer\.h\.([0-9]+)\.ln_1\.weight$": (
          r"layers.\1.ln_1.w",
          None,
      ),
      r"transformer\.h\.([0-9]+)\.ln_1\.bias$": (
          r"layers.\1.ln_1.b",
          None,
      ),
      r"transformer\.h\.([0-9]+)\.ln_2\.weight$": (
          r"layers.\1.ln_2.w",
          None,
      ),
      r"transformer\.h\.([0-9]+)\.ln_2\.bias$": (
          r"layers.\1.ln_2.b",
          None,
      ),
      r"transformer\.h\.([0-9]+)\.attn\.attention\.q_proj\.weight$": (
          r"layers.\1.attn.q_proj.kernel",
          ((1, 0), None),
      ),
      r"transformer\.h\.([0-9]+)\.attn\.attention\.k_proj\.weight$": (
          r"layers.\1.attn.k_proj.kernel",
          ((1, 0), None),
      ),
      r"transformer\.h\.([0-9]+)\.attn\.attention\.v_proj\.weight$": (
          r"layers.\1.attn.v_proj.kernel",
          ((1, 0), None),
      ),
      r"transformer\.h\.([0-9]+)\.attn\.attention\.out_proj\.weight$": (
          r"layers.\1.attn.out_proj.kernel",
          ((1, 0), None),
      ),
      r"transformer\.h\.([0-9]+)\.attn\.attention\.out_proj\.bias$": (
          r"layers.\1.attn.out_proj.bias",
          None,
      ),
      r"transformer\.h\.([0-9]+)\.mlp\.c_fc\.weight$": (
          r"layers.\1.mlp.c_fc.kernel",
          ((1, 0), None),
      ),
      r"transformer\.h\.([0-9]+)\.mlp\.c_fc\.bias$": (
          r"layers.\1.mlp.c_fc.bias",
          None,
      ),
      r"transformer\.h\.([0-9]+)\.mlp\.c_proj\.weight$": (
          r"layers.\1.mlp.c_proj.kernel",
          ((1, 0), None),
      ),
      r"transformer\.h\.([0-9]+)\.mlp\.c_proj\.bias$": (
          r"layers.\1.mlp.c_proj.bias",
          None,
      ),
  }
  if not cfg.use_tied_embedding:
    mapping[r"lm_head\.weight$"] = ("lm_head.kernel", ((1, 0), None))
  return mapping


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
) -> model_lib.GPTNeo:
  """Load tensors from the safetensors file and create a GPT-Neo model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.GPTNeo,
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
    return "lm_head.weight"
  return f"{lora_name}.weight".replace("layers.", "transformer.h.", 1).replace(
      ".attn.", ".attn.attention."
  )


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.GPTNeo,
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
