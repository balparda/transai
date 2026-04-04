# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Base AI library."""

from __future__ import annotations

import abc
import logging
import pathlib
from typing import Self, TypedDict, final

import llama_cpp
import lmstudio
import pydantic
from transcrypto.utils import base, saferandom

_LMSTUDIO_ROOT: pathlib.Path = pathlib.Path('~/.lmstudio/models/').expanduser().resolve()
DEFAULT_MODELS_ROOT: pathlib.Path | None = _LMSTUDIO_ROOT if _LMSTUDIO_ROOT.is_dir() else None

AI_CONTEXT_LENGTH = 32 * 1024  # 32k tokens, should be enough for the image and the tags
AI_MAX_CONTEXT = 2**24  # 16 million tokens, just a sanity check upper bound for validation
DEFAULT_VISION_MODEL = 'qwen3-vl-32b-instruct@Q8_0'
DEFAULT_TEXT_MODEL = 'qwen3-8b@Q8_0'
AI_MAX_SEED = 2**31 - 1
DEFAULT_GPU_RATIO = 0.8
DEFAULT_TEMPERATURE = 0.15
MAX_TEMPERATURE = 2.0


class Error(base.Error):
  """Base class for AI-related errors."""


class AIModelConfig(TypedDict):
  """Configuration for an AI model."""

  model_id: str  # model standardized key
  version: str  # `transai` library version, for compatibility checks and debugging
  model_path: pathlib.Path | None  # full path to the model
  clip_path: pathlib.Path | None  # full path to the CLIP file used for loading (if vision)
  seed: int | None  # seed, for reproducibility
  context: int  # context length (max)
  temperature: float  # sampling temperature for generation
  gpu_ratio: float  # GPU usage (0.0-1.0)
  gpu_layers: int  # number of layers offloaded to GPU (-1 for as many as possible)
  use_mmap: bool  # whether to use memory-mapped file loading (if supported)
  vision: bool  # can the model process images?
  tooling: bool  # can the model use tools (e.g. external APIs, code execution, etc.)?
  reasoning: bool  # can the model perform multi-step reasoning (e.g. chain-of-thought)?
  fp16: bool  # is the model loaded in fp16 precision (for debugging / analysis)?
  flash: bool  # flash attention enabled (if supported)
  spec_tokens: int | None  # number of tokens used for speculative decoding (if applicable)
  kv_cache: int | None  # GGML type for KV-cache keys/values


type AIModelMetadata = base.JSONDict  # metadata about the loaded model
type AIImageInput = bytes | pathlib.Path | str
type _SupportedModelObject = llama_cpp.Llama | lmstudio.LLM  # supported backends actual model type
type LoadedModel = tuple[AIModelConfig, AIModelMetadata, _SupportedModelObject]
type _LoadedModelsDict = dict[str, LoadedModel]
_LLM_REQUIRING_CLOSE_METHOD: tuple[type[_SupportedModelObject], ...] = (llama_cpp.Llama,)


class AIWorker(abc.ABC):
  """Abstract base class for AI worker."""

  def __init__(self) -> None:
    """Initialize the worker."""
    self._loaded_models: _LoadedModelsDict = {}

  @final
  def __enter__(self) -> Self:
    """Context manager entry, returns self.

    Returns:
      self, for use within the context

    """
    return self

  @final
  def __exit__(self, *args: object) -> None:
    """Context manager exit, closes any started sessions.

    Args:
      *args: standard context manager args (exc_type, exc_value, traceback), ignored here

    """
    self.Close()

  def Close(self) -> None:
    """Close any started sessions."""
    logging.info('Closing model objects')
    for model_id, (_, _, model) in self._loaded_models.items():
      if isinstance(model, _LLM_REQUIRING_CLOSE_METHOD):
        logging.info(f'Releasing model {model_id!r}')
        model.close()  # type: ignore[union-attr]
    self._loaded_models.clear()

  @final
  def LoadModel(self, config: AIModelConfig, /) -> tuple[AIModelConfig, AIModelMetadata]:
    """Load the model with the given configuration.

    Args:
      config: AIModelConfig with loading parameters, `model_id` must be provided; the other fields
          may be ignored or overridden by the caller; the loading implementation should fill
          in any missing fields with the actual values used for loading

    Returns:
      (
        AIModelConfig: with the actual loading configuration used
            (including any inferred or overridden fields),
        ModelMetadata: metadata about the loaded model,
      )

    """
    new_model: LoadedModel = self._Load(self._ConfigSeed(config))
    config = new_model[0]
    self._loaded_models[config['model_id']] = (
      new_model[0].copy(),
      new_model[1].copy(),
      new_model[2],
    )
    logging.info(f'Loaded model {config["model_id"]!r}: {new_model[0]!r} / {new_model[1]!r}')
    return (new_model[0], new_model[1])

  @abc.abstractmethod
  def _Load(self, config: AIModelConfig, /) -> LoadedModel:
    """Load the model with the given configuration.

    Args:
      config: AIModelConfig with loading parameters, `model_id` must be provided; the other fields
          may be ignored or overridden by the caller; the loading implementation should fill
          in any missing fields with the actual values used for loading

    Returns:
      (
        AIModelConfig: with the actual loading configuration used
            (including any inferred or overridden fields),
        ModelMetadata: metadata about the loaded model,
        _SupportedModelObject: the loaded model instance (e.g. llama_cpp.Llama or lmstudio.LLM)
      )

    Raises:
      Error: if loading fails for any reason (e.g. invalid config, model not found, etc)

    """

  @final
  def _ConfigSeed(self, config: AIModelConfig, /) -> AIModelConfig:  # noqa: PLR6301
    """Fill in seed configuration field for the AI model; check validity of some fields.

    Args:
      config: AIModelConfig with loading parameters, `model_id` must be provided; the other fields
          may be ignored or overridden by the caller

    Returns:
      AIModelConfig with the seed field correctly filled in

    Raises:
      Error: if any of the config values are invalid (e.g. gpu out of range, seed out of range, etc)

    """
    # check id
    model_id: str = config['model_id'].strip().lower()
    if not model_id:
      raise Error('AIModelConfig.model_id must be a non-empty string')
    # check context
    if not 16 <= config['context'] <= AI_MAX_CONTEXT:  # noqa: PLR2004
      raise Error(f'16<=AIModelConfig.context<={AI_MAX_CONTEXT}, got {config["context"]}')
    # check temperature
    if not 0.0 <= config['temperature'] <= MAX_TEMPERATURE:
      raise Error(f'0.0<=AIModelConfig.temperature<={MAX_TEMPERATURE}, got {config["temperature"]}')
    # check GPU
    if not 0.1 <= config['gpu_ratio'] <= 1.0:  # noqa: PLR2004
      raise Error(f'AIModelConfig.gpu_ratio must be between .1 and 1.0, got {config["gpu_ratio"]}')
    # we can't retrieve the seed later, so we inject here if not given
    actual_seed: int = saferandom.RandBits(31) if config['seed'] is None else config['seed']
    if not 1 <= actual_seed <= AI_MAX_SEED:
      raise Error(f'seed must be between 1 and 2^31-1, got {actual_seed}')
    new_config: AIModelConfig = config.copy()
    new_config.update({'model_id': model_id, 'seed': actual_seed})
    return new_config

  @final
  def ModelCall[T: pydantic.BaseModel | str](
    self,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    output_format: type[T],
    /,
    *,
    images: list[AIImageInput] | None = None,
  ) -> T:
    """Make a call to the model.

    Args:
      model_id: the standardized model key (not the path) of the model to call; must be one of the
          models previously loaded with LoadModel()
      system_prompt: the system prompt to provide context or instructions to the model
      user_prompt: the user prompt containing the actual query or request for the model
      output_format: optional pydantic model class or `str` to parse the output into;
          if not given, the raw string output from the model will be returned
      images (default=None): optional list of images to send as input, either as bytes or file
          paths; only supported if the model has vision capability

    Returns:
      the model output, either as a raw string or parsed into the given `output_format` class

    Raises:
      Error: if the `model_id` is not found, if the model does not support the given inputs, or
          if there is any error calling the model

    """
    if model_id not in self._loaded_models:
      raise Error(f'Model {model_id!r} not loaded; call LoadModel() first')
    return self._Call(
      self._loaded_models[model_id], system_prompt, user_prompt, output_format, images=images
    )

  @abc.abstractmethod
  def _Call[T: pydantic.BaseModel | str](
    self,
    model: LoadedModel,
    system_prompt: str,
    user_prompt: str,
    output_format: type[T],
    /,
    *,
    images: list[AIImageInput] | None = None,
  ) -> T:
    """Make a call to the model.

    Args:
      model: the loaded model instance to call; one of the models previously loaded with _Load()
      system_prompt: the system prompt to provide context or instructions to the model
      user_prompt: the user prompt containing the actual query or request for the model
      output_format: optional pydantic model class or `str` to parse the output into;
          if not given, the raw string output from the model will be returned
      images (default=None): optional list of images to send as input, either as bytes or file
          paths; only supported if the model has vision capability

    Returns:
      the model output, either as a raw string or parsed into the given `output_format` class

    Raises:
      Error: if the model does not support the given inputs, or if there is any error calling

    """
