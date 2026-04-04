# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""LM Studio (LMS) AI library."""

from __future__ import annotations

import logging
from typing import cast

import lmstudio
import pydantic

from transai.core import ai


class Error(ai.Error):
  """LM Studio AI library error."""


class LMStudioWorker(ai.AIWorker):
  """AI worker implementation using LMStudio."""

  def __init__(self) -> None:
    """Connect to LM Studio API server, do some checks, unload existing models.

    Raises:
      Error: if no LM Studio API server instance is found on the default local ports, or
          if the found API server is not on a loopback address (potential security risk)

    """
    super().__init__()
    logging.info('Starting LM Studio (lms) connection, configs, checks...')
    if (api_host := lmstudio.Client.find_default_local_api_host()) is None:
      raise Error('No LM Studio API server instance found on any of the default local ports')
    if not api_host.startswith('127.0.0.1:') and not api_host.startswith('localhost:'):
      # we are meant to be strictly local
      raise Error(
        f'LM Studio API server found at {api_host} which is not a loopback address: '
        'this may be a security risk if the server is not properly firewalled'
      )
    logging.info(f'LM Studio @ {api_host}')
    lmstudio.set_sync_api_timeout(None)  # None is infinite timeout
    self._api_host: str = api_host
    self._client: lmstudio.Client = lmstudio.Client(api_host)
    # unload all currently loaded LLMs to free VRAM/RAM before loading our model;
    # without this, loading a large model on top of others will likely exhaust system resources
    for loaded in self._client.llm.list_loaded():
      logging.info(f'Unloading existing model {loaded.identifier!r} to free resources')
      self._client.llm.unload(loaded.identifier)

  def Close(self) -> None:
    """Close any started sessions."""
    logging.info(f'Closing LM Studio client connection @ {self._api_host}')
    self._client.close()
    super().Close()

  def _Load(self, config: ai.AIModelConfig, /) -> ai.LoadedModel:
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
    if config['model_path'] is not None or config['clip_path'] is not None:
      logging.warning('AIModelConfig.model_path/clip_path are ignored by LMStudioWorker')
    if config['k_cache'] is not None or config['v_cache'] is not None:
      logging.warning('AIModelConfig.k_cache/v_cache are ignored by LMStudioWorker')
    if config['flash'] or config['gpu_layers'] != -1 or config['spec_tokens'] is not None:
      logging.warning('AIModelConfig.flash/gpu_layers/spec_tokens are ignored by LMStudioWorker')
    if config['reasoning']:
      # TODO: check reasoning capabilities when lmstudio supports it in the future
      raise Error(f'AIModelConfig.reasoning is not supported by LMStudioWorker: {config!r}')
    # create LMStudio config
    load_config = lmstudio.LlmLoadModelConfigDict(
      # <https://lmstudio.ai/docs/typescript/api-reference/llm-load-model-config>
      # The size of the context length in number of tokens.
      # This will include both the prompts and the responses.
      contextLength=config['context'],
      # Random seed value for model initialization to ensure reproducible outputs.
      seed=config['seed'],
      # Attempts to use memory-mapped (mmap) file access when loading the model.
      # Memory mapping can improve initial load times by mapping model files directly from
      # disk to memory, allowing the operating system to handle paging. This is particularly
      # beneficial for quick startup, but may reduce performance if the model is larger
      # than available system RAM, causing frequent disk access.
      tryMmap=config['use_mmap'],
      # This option significantly reduces memory usage during inference by using 16-bit floating
      # point numbers instead of 32-bit for the attention cache. While this may slightly reduce
      # numerical precision, the output quality impact is generally minimal for most applications.
      useFp16ForKVCache=config['fp16'],
      # This option determines the precision level used to store the key component of the
      # attention mechanism's cache. Lower precision values (e.g., 4-bit or 8-bit quantization)
      # significantly reduce memory usage during inference but may slightly impact output quality.
      # The effect varies between different models, with some being more robust to quantization
      # than others.
      llamaKCacheQuantizationType=None,  # we wanted False, but there is a bug in lmstudio
      llamaVCacheQuantizationType=None,  # we wanted False, but there is a bug in lmstudio
      # How to distribute the work to your GPUs.
      gpu={'ratio': config['gpu_ratio']},
      # Flash Attention is an efficient implementation that reduces memory usage and speeds up
      # generation by optimizing how attention mechanisms are computed. This can significantly
      # improve performance on compatible hardware, especially for longer sequences.
      flashAttention=config['flash'],
    )
    # load model
    logging.info(f'Loading model {config["model_id"]!r}')
    model: lmstudio.LLM = self._client.llm.load_new_instance(config['model_id'], config=load_config)
    # do basic model validation and return if OK
    model_info = model.get_info()
    if not isinstance(model_info, lmstudio.LlmInstanceInfo):
      raise Error(f'Model {config["model_id"]} not a valid LLM instance')
    model_key: str = model_info.model_key.strip()
    if config['vision'] and not model_info.vision:
      raise Error(f'Model {model_key} not a vision model')
    if config['tooling'] and not model_info.trained_for_tool_use:
      raise Error(f'Model {model_key} not trained for tool use')
    if (n_ctx := model.get_context_length()) < config['context']:
      raise Error(
        f'Model {model_key} has insufficient context length '
        f'({n_ctx}/{model_info.max_context_length}) for '
        'auto-tagging: you should select a vision model with at least 32k context length'
      )
    new_config: ai.AIModelConfig = config.copy()
    new_config.update(
      {
        'model_id': model_key,
        'vision': model_info.vision,
        'tooling': model_info.trained_for_tool_use,
      }
    )
    return (new_config, cast('ai.AIModelMetadata', model_info.to_dict()), model)

  def _Call[T: pydantic.BaseModel | str](  # noqa: PLR6301
    self,
    model: ai.LoadedModel,
    system_prompt: str,
    user_prompt: str,
    output_format: type[T],
    /,
    *,
    images: list[ai.AIImageInput] | None = None,
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
    model_config: ai.AIModelConfig = model[0]
    lm_model: lmstudio.LLM = cast('lmstudio.LLM', model[2])
    model_id: str = model_config['model_id']
    config = lmstudio.LlmPredictionConfigDict(
      # Number of tokens to predict at most.
      # If set to false, the model will predict as many tokens as it wants.
      maxTokens=model_config['context'],
      # The temperature parameter for the prediction model. A higher value makes the predictions
      # more random, while a lower value makes the predictions more deterministic.
      # The value should be between 0 and 1.
      temperature=model_config['temperature'],
    )
    chat = lmstudio.Chat(system_prompt)
    chat.add_user_message(
      user_prompt,
      images=[lmstudio.prepare_image(b) for b in images] if images else None,  # type: ignore[arg-type]
    )
    # call the model
    logging.debug(f'Calling AI with config {config!r} and chat {chat!r}')
    try:
      result: lmstudio.PredictionResult = lm_model.respond(
        chat,
        config=config,
        response_format=None if output_format is str else output_format,  # type: ignore[arg-type]
      )
    except lmstudio.LMStudioServerError as err:
      raise Error(f'Error calling model {model_id!r}: {err}') from err
    # log and check results
    logging.debug('Predicted tokens: %d', result.stats.predicted_tokens_count)
    logging.debug('Time to first token (seconds): %f', result.stats.time_to_first_token_sec)
    if result.stats.stop_reason != 'eosFound':
      raise Error(f'Unexpected stop reason {result.stats.stop_reason!r} while generating concilium')
    # parse and return the verdict
    return result.content if output_format is str else output_format.model_validate(result.parsed)  # type: ignore[return-value,attr-defined]
