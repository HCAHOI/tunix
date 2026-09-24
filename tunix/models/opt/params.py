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

"""Utils for loading and converting OPT PT weights."""

import contextlib
import functools
import pathlib
import shutil
import tempfile

import jax
import jax.numpy as jnp
import safetensors
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.opt import model as model_lib

# The original facebook/opt checkpoints prefix keys with ``decoder.``; models
# re-saved from ``OPTForCausalLM`` use ``model.decoder.``. Accept either.
_P = r"(?:model\.)?decoder\."


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule))."""
  # PyTorch nn.Linear weights are [out, in]; nnx.Linear kernels are [in, out],
  # so linear weights transpose with permute (1, 0). Embeddings load as-is.
  # The LM head is tied to embed_tokens (no separate weight in the checkpoint).
  mapping = {
      _P + r"embed_tokens\.weight$": ("embedder.embed_tokens", None),
      _P + r"embed_positions\.weight$": ("embedder.embed_positions", None),
  }
  if not cfg.use_tied_embedding:
    mapping[r"lm_head\.weight$"] = ("lm_head.kernel", ((1, 0), None))
  if cfg.has_projection:
    mapping[_P + r"project_in\.weight$"] = (
        "embedder.project_in.kernel",
        ((1, 0), None),
    )
    mapping[_P + r"project_out\.weight$"] = (
        "embedder.project_out.kernel",
        ((1, 0), None),
    )
  if cfg.do_layer_norm_before:
    mapping[_P + r"final_layer_norm\.weight$"] = ("final_norm.w", None)
    mapping[_P + r"final_layer_norm\.bias$"] = ("final_norm.b", None)

  layer = _P + r"layers\.([0-9]+)\."
  mapping.update({
      layer
      + r"self_attn\.q_proj\.weight$": (
          r"layers.\1.attn.q_proj.kernel",
          ((1, 0), None),
      ),
      layer
      + r"self_attn\.q_proj\.bias$": (r"layers.\1.attn.q_proj.bias", None),
      layer
      + r"self_attn\.k_proj\.weight$": (
          r"layers.\1.attn.k_proj.kernel",
          ((1, 0), None),
      ),
      layer
      + r"self_attn\.k_proj\.bias$": (r"layers.\1.attn.k_proj.bias", None),
      layer
      + r"self_attn\.v_proj\.weight$": (
          r"layers.\1.attn.v_proj.kernel",
          ((1, 0), None),
      ),
      layer
      + r"self_attn\.v_proj\.bias$": (r"layers.\1.attn.v_proj.bias", None),
      layer
      + r"self_attn\.out_proj\.weight$": (
          r"layers.\1.attn.out_proj.kernel",
          ((1, 0), None),
      ),
      layer
      + r"self_attn\.out_proj\.bias$": (r"layers.\1.attn.out_proj.bias", None),
      layer
      + r"self_attn_layer_norm\.weight$": (
          r"layers.\1.self_attn_layer_norm.w",
          None,
      ),
      layer
      + r"self_attn_layer_norm\.bias$": (
          r"layers.\1.self_attn_layer_norm.b",
          None,
      ),
      layer + r"fc1\.weight$": (r"layers.\1.mlp.fc1.kernel", ((1, 0), None)),
      layer + r"fc1\.bias$": (r"layers.\1.mlp.fc1.bias", None),
      layer + r"fc2\.weight$": (r"layers.\1.mlp.fc2.kernel", ((1, 0), None)),
      layer + r"fc2\.bias$": (r"layers.\1.mlp.fc2.bias", None),
      layer
      + r"final_layer_norm\.weight$": (r"layers.\1.final_layer_norm.w", None),
      layer
      + r"final_layer_norm\.bias$": (r"layers.\1.final_layer_norm.b", None),
  })
  return mapping


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
) -> model_lib.OPT:
  """Load tensors from the safetensors file and create an OPT model."""
  with _safetensors_directory(file_dir) as weights_dir:
    return safetensors_loader.load_and_create_model(
        file_dir=weights_dir,
        model_class=model_lib.OPT,
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
    lora_model: model_lib.OPT,
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
    ValueError: The output would overwrite the source, or the input is sharded.
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
    with safetensors.safe_open(
        str(pathlib.Path(weights_dir) / "model.safetensors"), framework="numpy"
    ) as checkpoint:
      prefix = (
          "" if "decoder.embed_tokens.weight" in checkpoint.keys() else "model."
      )
    safetensors_saver.save_lora_merged_model_as_safetensors(
        local_model_path=weights_dir,
        output_dir=output_dir,
        lora_model=lora_model,
        rank=rank,
        alpha=alpha,
        state_key_transform_fn=functools.partial(
            _state_key_to_safetensors_key, prefix=prefix
        ),
        transpose_rules={"weight": (1, 0)},
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


def _state_key_to_safetensors_key(lora_name: str, *, prefix: str) -> str:
  """Maps a LoRA module path, preserving the checkpoint's decoder prefix."""
  if lora_name == "lm_head":
    return "lm_head.weight"
  name = lora_name.removeprefix("embedder.")
  name = name.replace(".attn.", ".self_attn.").replace(".mlp.", ".")
  return f"{prefix}decoder.{name}.weight"
