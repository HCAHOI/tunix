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

import contextlib
import pathlib
import shutil
import tempfile

import jax
import jax.numpy as jnp
import safetensors
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.gpt_neox import model as model_lib
from tunix.utils import torch_utils


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


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model,
    rank: int,
    alpha: float,
):
  """Merges LoRA weights and copies checkpoint metadata to output_dir.

  Args:
    local_model_path: Directory with one model.safetensors or pytorch_model.bin.
    output_dir: Destination directory for merged weights and metadata.
    lora_model: NNX model with LoRA adapters and a model configuration.
    rank: LoRA rank.
    alpha: LoRA scaling factor.

  Raises:
    ValueError: The output would overwrite the source, the input is sharded,
      or a linear weight requires an unsupported transformation.
    ImportError: PyTorch conversion dependencies are unavailable.
  """
  source = pathlib.Path(local_model_path).resolve()
  output = pathlib.Path(output_dir).resolve()
  if source == output or output in source.parents:
    raise ValueError("Merged output must not overwrite the source checkpoint")
  with _safetensors_directory(local_model_path) as weights_dir:
    if not (pathlib.Path(weights_dir) / "model.safetensors").is_file():
      raise ValueError(
          "LoRA merge requires a single model.safetensors or "
          "pytorch_model.bin checkpoint; sharded export is not supported."
      )
    key_mapping = _get_key_and_transform_mapping(lora_model.config)
    _save_lora_merged_model(
        weights_dir, output_dir, lora_model, rank, alpha, key_mapping
    )
  if weights_dir != local_model_path:
    for file in pathlib.Path(local_model_path).iterdir():
      if (
          file.is_file()
          and file.suffix not in (".bin", ".safetensors")
          and not file.name.endswith(".index.json")
      ):
        shutil.copy2(file, pathlib.Path(output_dir) / file.name)


@contextlib.contextmanager
def _safetensors_directory(file_dir: str):
  """Yields a safetensors directory without modifying the source checkpoint.

  Args:
    file_dir: Checkpoint directory or GCS path. Local PyTorch checkpoints must
      contain a single pytorch_model.bin file.

  Yields:
    The input directory for safetensors weights, or a temporary directory for
    converted PyTorch weights. The temporary directory exists only within the
    context manager.

  Raises:
    ValueError: The input is a sharded PyTorch checkpoint.
    ImportError: Conversion requires the optional legacy dependencies.
  """
  path = pathlib.Path(file_dir).expanduser()
  if file_dir.startswith("gs://") or any(path.glob("*.safetensors")):
    yield file_dir
    return
  if (path / "pytorch_model.bin.index.json").is_file():
    raise ValueError(
        "Sharded PyTorch checkpoints are not supported. Provide a single "
        "pytorch_model.bin or safetensors weights."
    )
  file = path / "pytorch_model.bin"
  if not file.is_file():
    yield file_dir
    return
  try:
    # Keep PyTorch optional when loading safetensors checkpoints.
    from safetensors import torch as safetensors_torch  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
  except ImportError as error:
    raise ImportError(
        "This checkpoint contains PyTorch .bin weights. Install "
        "google-tunix[legacy] or provide converted safetensors weights."
    ) from error
  with tempfile.TemporaryDirectory(prefix="tunix-converted-") as directory:
    weights = torch.load(file, map_location="cpu", weights_only=True)
    safetensors_torch.save_file(
        {k: v.detach().contiguous().clone() for k, v in weights.items()},
        str(pathlib.Path(directory) / "model.safetensors"),
    )
    yield directory


def _save_lora_merged_model(
    local_model_path, output_dir, lora_model, rank, alpha, key_mapping
):
  """Resolves checkpoint key names and delegates merging to the Tunix saver."""
  reverse = {}
  with safetensors.safe_open(
      str(pathlib.Path(local_model_path) / "model.safetensors"),
      framework="numpy",
  ) as checkpoint:
    for key in checkpoint.keys():
      try:
        destination, transform = torch_utils.torch_key_to_jax_key(
            key_mapping, key
        )
      except ValueError:
        continue
      if destination.endswith(".kernel"):
        if transform != ((1, 0), None):
          raise ValueError(
              f"Unsupported linear weight transform: {key}: {transform}"
          )
        reverse[destination.removesuffix(".kernel")] = key
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=reverse.__getitem__,
      transpose_rules={"weight": (1, 0)},
  )
