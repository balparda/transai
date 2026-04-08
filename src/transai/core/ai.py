# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Base AI library."""

from __future__ import annotations

import abc
import collections.abc
import concurrent.futures
import dataclasses
import functools
import json
import logging
import pathlib
import re
from typing import Any, Self, TypedDict, final

import llama_cpp
import lmstudio
import pydantic
from transcrypto.core import hashes, modmath
from transcrypto.utils import base, human, saferandom

from transai import __version__

_LMSTUDIO_ROOT: pathlib.Path = pathlib.Path('~/.lmstudio/models/').expanduser().resolve()
DEFAULT_MODELS_ROOT: pathlib.Path | None = _LMSTUDIO_ROOT if _LMSTUDIO_ROOT.is_dir() else None

DEFAULT_TIMEOUT: float = 5 * 60.0  # 5 minutes, just a reasonable default
AI_CONTEXT_LENGTH = 32 * 1024  # 32k tokens, should be enough for the image and the tags
AI_MAX_CONTEXT = 2**24  # 16 million tokens, just a sanity check upper bound for validation
DEFAULT_VISION_MODEL = 'qwen3-vl-32b-instruct@Q8_0'
DEFAULT_TEXT_MODEL = 'qwen3-8b@Q8_0'
AI_MAX_SEED = 2**31 - 1  # another good thing is that this is prime!
assert modmath.IsPrime(AI_MAX_SEED), 'AI_MAX_SEED prime for better seed distribution and properties'  # noqa: S101
DEFAULT_GPU_RATIO = 0.8
DEFAULT_TEMPERATURE = 0.15
MAX_TEMPERATURE = 2.0

RE_THINK: re.Pattern[str] = re.compile(r'<think>(.*?)</think>', flags=re.DOTALL)
RE_TOOL_CALL: re.Pattern[str] = re.compile(r'<tool_call>(.*?)</tool_call>', flags=re.DOTALL)


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


@functools.total_ordering
@dataclasses.dataclass(kw_only=True, slots=True)
class LoadedModel:
  """Loaded AI model, with its configuration and metadata.

  Attributes:
    model_id: the standardized model key (not the path) of the loaded model
    seed_state: opaque bytes hash SHA256 is the state; will change on every call; 32 bytes long
    config: the AIModelConfig used for loading the model, with all fields filled in and standardized
    metadata: metadata about the loaded model (e.g. actual model path, CLIP path, etc)
    model: the actual loaded model object (e.g. llama_cpp.Llama or lmstudio.LLM instance)

  """

  model_id: str
  seed_state: bytes  # 32 bytes
  config: AIModelConfig
  metadata: AIModelMetadata
  model: _SupportedModelObject

  def __lt__(self, other: LoadedModel) -> bool:
    """Less than. Makes sortable (b/c base class already defines __eq__).

    Args:
      other (LoadedModel): other to compare against

    Returns:
      bool: True if this LoadedModel is less than the other, False otherwise.

    """
    return self.model_id < other.model_id


type AIModelMetadata = base.JSONDict  # metadata about the loaded model
type AIImageInput = bytes | pathlib.Path | str
type AnyCallable = collections.abc.Callable[..., Any]
type AIToolInput = str | AnyCallable
type _SupportedModelObject = llama_cpp.Llama | lmstudio.LLM  # supported backends actual model type
type _LoadedModelsDict = dict[str, LoadedModel]
_LLM_REQUIRING_CLOSE_METHOD: tuple[type[_SupportedModelObject], ...] = (llama_cpp.Llama,)


def MakeAIModelConfig(**overrides: object) -> AIModelConfig:
  """Create a valid default AIModelConfig.

  Returns:
    AIModelConfig with all required fields set to valid defaults, and overrides applied

  """
  base: AIModelConfig = {
    'model_id': DEFAULT_TEXT_MODEL,
    'version': __version__,
    'model_path': None,
    'clip_path': None,
    'seed': None,
    'context': AI_CONTEXT_LENGTH,
    'temperature': DEFAULT_TEMPERATURE,
    'gpu_ratio': DEFAULT_GPU_RATIO,
    'gpu_layers': -1,
    'vision': False,
    'tooling': False,  # TODO: implement tooling support
    'reasoning': False,
    'fp16': False,
    'use_mmap': True,
    'flash': True,
    'spec_tokens': None,
    'kv_cache': None,
  }
  base.update(overrides)  # type: ignore[typeddict-item]
  return base


class AIWorker(abc.ABC):
  """Abstract base class for AI worker."""

  def __init__(self, /, *, timeout: float | None = DEFAULT_TIMEOUT) -> None:
    """Initialize the worker.

    Args:
      timeout (default=DEFAULT_TIMEOUT): optional timeout in seconds for model loading and calls;
          if not given, defaults to DEFAULT_TIMEOUT; can be set to None for no timeout

    """
    self._loaded_models: _LoadedModelsDict = {}
    self._timeout: float | None = timeout
    logging.info(f'AI timeout set to {human.HumanizedSeconds(timeout) if timeout else "None"}')

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
    for model_id, loaded in self._loaded_models.items():
      if isinstance(loaded.model, _LLM_REQUIRING_CLOSE_METHOD):
        logging.info(f'Releasing model {model_id!r}')
        loaded.model.close()  # type: ignore[union-attr]
    self._loaded_models.clear()

  @final
  def _RunWithTimeout[T](self, func: collections.abc.Callable[[], T], /, *, description: str) -> T:
    """Run a callable with a timeout, raising Error if it takes too long.

    Args:
      func: a zero-argument callable to run; bind any arguments before passing
      description: human-readable description used in the timeout error message

    Returns:
      the return value of `func`

    Raises:
      Error: if the operation exceeds `self._timeout` seconds

    """
    if self._timeout is None:
      return func()
    pool: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
      max_workers=1
    )
    future: concurrent.futures.Future[T] = pool.submit(func)
    try:
      result: T = future.result(timeout=self._timeout)
    except concurrent.futures.TimeoutError as err:
      pool.shutdown(wait=False, cancel_futures=True)
      raise Error(f'{description} timed out after {human.HumanizedSeconds(self._timeout)}') from err
    else:
      pool.shutdown(wait=False)
      return result

  @final
  def _RegisterModel(self, model: LoadedModel, /) -> None:
    """Register a loaded model in the worker's internal state.

    This should be called by the subclass implementations after successfully loading a model,
    to keep track of the loaded models and their configurations.

    Args:
      model: the loaded model tuple to register

    Raises:
      Error: on error

    """
    config: AIModelConfig = self._ConfigSeed(model.config)
    if not config['seed'] or config['seed'] <= 1:  # for safety, but should never happen
      raise Error('Loaded model config must have a seed to be registered')
    self._loaded_models[config['model_id']] = LoadedModel(
      model_id=config['model_id'],
      seed_state=hashes.Hash256(base.IntToBytes(config['seed'])),
      config=config,
      metadata=model.metadata.copy(),
      model=model.model,
    )
    logging.info(f'Registered model {config["model_id"]!r}: {config!r} / {model.metadata!r}')

  @final
  def LoadModel(
    self, config: AIModelConfig, /, *, force: bool = False, ignore_quant: bool = True
  ) -> tuple[AIModelConfig, AIModelMetadata]:
    """Load the model with the given configuration.

    Args:
      config: AIModelConfig with loading parameters, `model_id` must be provided; the other fields
          may be ignored or overridden by the caller; the loading implementation should fill
          in any missing fields with the actual values used for loading
      force (default=False): whether to force reload the model even if it is already loaded
      ignore_quant (default=True): whether to ignore quantization part of `model_id`

    Returns:
      (
        AIModelConfig: with the actual loading configuration used
            (including any inferred or overridden fields),
        ModelMetadata: metadata about the loaded model,
      )

    Raises:
      Error: on loading errors, including if the operation exceeds the configured timeout

    """
    if not force and config['seed'] is not None:
      force = True  # if a seed is specified, we have to force reload to apply it
      logging.info(f'Seed {config["seed"]} specified in config, forcing model reload to apply')
    config = self._ConfigSeed(config)  # standardize 'model_id' and 'seed'
    if not config['seed'] or config['seed'] <= 1:  # for safety, but should never happen
      raise Error('Config must have a seed to be loaded')
    # if the exact model is already loaded and we're not forcing, return it
    if not force and config['model_id'] in self._loaded_models:
      logging.info(f'Model {config["model_id"]!r} already loaded, returning existing instance')
      existing: LoadedModel = self._loaded_models[config['model_id']]
      return (existing.config.copy(), dict(existing.metadata))
    # if ignoring quantization and the generic version of the model is already loaded, return it
    if (
      not force
      and ignore_quant
      and (reduced := config['model_id'].rsplit('@', 1)[0]) in self._loaded_models
    ):
      logging.info(f'Model {config["model_id"]!r} found as generic quantized version')
      existing = self._loaded_models[reduced]
      return (existing.config.copy(), dict(existing.metadata))
    # otherwise, we need to load the model which will be done by the subclass implementations
    try:
      new_model: LoadedModel = self._RunWithTimeout(
        lambda: self._LoadNew(config),
        description=f'Loading model {config["model_id"]!r}',
      )
    except Error:
      raise  # re-raise timeout errors (and other Error subclasses) as-is
    except Exception as err:
      # convert generic exceptions to our Error type for better error handling and debugging
      raise Error(f'Error loading model {config["model_id"]!r}') from err
    self._loaded_models[new_model.model_id] = new_model
    logging.info(f'Loaded {new_model.model_id!r}: {new_model.config!r} / {new_model.metadata!r}')
    return (new_model.config.copy(), dict(new_model.metadata))

  @abc.abstractmethod
  def _LoadNew(self, config: AIModelConfig, /) -> LoadedModel:
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
    if config['seed'] == -1:
      config['seed'] = None  # just for safety, but shouldn't happen
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
    tools: list[AIToolInput] | None = None,
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
      tools (default=None): optional list of tools (methods) to use during the call;
          only supported if the model has tool capability; mandates str `output_format`;
          also make sure the methods are all typed and have proper docstrings for best results

    Returns:
      the model output, either as a raw string or parsed into the given `output_format` class

    Raises:
      Error: if the `model_id` is not found, if the model does not support the given inputs,
          if there is any error calling the model, or if the call exceeds the configured timeout

    """
    if model_id not in self._loaded_models:
      raise Error(f'Model {model_id!r} not loaded; call LoadModel() first')
    # get loaded model and iterate seed state
    loaded: LoadedModel = self._loaded_models[model_id]
    if images and not loaded.config['vision']:
      raise Error(f'Model {model_id!r} does not support vision inputs, but images were provided')
    if tools and not loaded.config['tooling']:
      raise Error(f'Tools provided but model {model_id!r} not trained for tool use')
    if tools and output_format is not str:
      # TODO: see if llama.cpp can handle this case...
      raise Error(f'Model {model_id!r} asked to use tools and return non-str output: unsupported')
    new_seed: int = 0
    while new_seed <= 1:  # just for safety, but should never (rarely) happen
      loaded.seed_state = hashes.Hash256(loaded.seed_state)  # S <- SHA256(S)
      new_seed = base.BytesToInt(loaded.seed_state) % AI_MAX_SEED  # (AI_MAX_SEED is prime)
    logging.info(f'Calling {model_id!r} @{new_seed} ({loaded.seed_state.hex()})')
    try:
      return self._RunWithTimeout(
        lambda: self._Call(
          loaded,
          system_prompt,
          user_prompt,
          output_format,
          new_seed,
          images=images,
          tools=[_GetCallable(t) for t in tools] if tools else None,
        ),
        description=f'Calling model {model_id!r}',
      )
    except json.JSONDecodeError as err:
      raise Error(f'Model {model_id!r} returned invalid JSON output') from err
    except Error:
      raise  # re-raise timeout errors (and other Error subclasses) as-is
    except Exception as err:
      # convert generic exceptions to our Error type for better error handling and debugging
      raise Error(f'Error calling model {model_id!r}') from err

  @abc.abstractmethod
  def _Call[T: pydantic.BaseModel | str](
    self,
    model: LoadedModel,
    system_prompt: str,
    user_prompt: str,
    output_format: type[T],
    call_seed: int,
    /,
    *,
    images: list[AIImageInput] | None = None,
    tools: list[AnyCallable] | None = None,
  ) -> T:
    """Make a call to the model.

    Args:
      model: the loaded model instance to call; one of the models previously loaded with _Load()
      system_prompt: the system prompt to provide context or instructions to the model
      user_prompt: the user prompt containing the actual query or request for the model
      output_format: optional pydantic model class or `str` to parse the output into;
          if not given, the raw string output from the model will be returned
      call_seed: the pre-computed seed to use for this call, derived from the model's seed state
      images (default=None): optional list of images to send as input, either as bytes or file
          paths; only supported if the model has vision capability
      tools (default=None): optional list of tools (methods) to use during the call;
          only supported if the model has tool capability; mandates str `output_format`;
          also make sure the methods are all typed and have proper docstrings for best results

    Returns:
      the model output, either as a raw string or parsed into the given `output_format` class

    Raises:
      Error: if the model does not support the given inputs, or if there is any error calling

    """


def _GetCallable(tool: AIToolInput) -> AnyCallable:
  """Convert a tool input (string or callable) into a callable function.

  Args:
    tool: the tool input, either a callable function or a string; the string should be a fully
        qualified name of a function, like e.g. 'math.gcd' or 'os.path.join', anything that
        can be imported and resolved to a callable function from this package

  Returns:
    a callable function corresponding to the tool input

  Raises:
    Error: if the tool input is a string but cannot be converted to a callable function

  """
  # if it's already a callable, just return it
  if callable(tool):
    return tool
  # otherwise, it should be a string that we need to resolve to a callable function
  func: Any
  try:
    module_path, func_name = tool.rsplit('.', 1)
    # load module and get function; this will raise an exception if it fails
    module = __import__(module_path, fromlist=[func_name])
    func = getattr(module, func_name)
  except (ImportError, AttributeError) as err:
    raise Error(f'Error resolving tool name {tool!r} to a callable function') from err
  if not func or not callable(func):
    raise Error(f'Tool {tool!r} resolved to {func!r} which is not callable')
  return func  # type: ignore[no-any-return]
