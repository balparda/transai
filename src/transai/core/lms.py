# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""LM Studio (LMS) AI library."""

from __future__ import annotations

import logging
from typing import cast

import lmstudio
import pydantic
from transcrypto.core import hashes
from transcrypto.utils import base

from transai.core import ai


class Error(ai.Error):
  """LM Studio AI library error."""


class LMStudioWorker(ai.AIWorker):
  """AI worker implementation using LMStudio."""

  def __init__(self, free_resources: bool = True) -> None:
    """Connect to LM Studio API server, do some checks, unload existing models.

    Args:
      free_resources: whether to unload all currently loaded LLMs to free VRAM/RAM before loading
          any models; without this, loading a large model on top of others will likely exhaust
          system resources, but setting this to False may be useful if you want to speed up

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
      if free_resources:
        logging.info(f'Unloading existing model {loaded.identifier!r} to free resources')
        self._client.llm.unload(loaded.identifier)
      else:
        info: ai.LoadedModel = _ExtractModelInfo(
          loaded, ai.MakeAIModelConfig(model_id=loaded.identifier)
        )
        self._RegisterModel(info)

  def Close(self) -> None:
    """Close any started sessions."""
    logging.info(f'Closing LM Studio client connection @ {self._api_host}')
    self._client.close()
    super().Close()

  def _LoadNew(self, config: ai.AIModelConfig, /) -> ai.LoadedModel:
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
    if config['kv_cache'] is not None:
      logging.warning('AIModelConfig.kv_cache is ignored by LMStudioWorker')
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
      # ATTENTION: my tests have shown that the seed value does NOT guarantee reproducibility in LMS
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
    # do basic model validation and return if OK
    return _ExtractModelInfo(
      self._client.llm.load_new_instance(config['model_id'], config=load_config), config
    )

  def _Call[T: pydantic.BaseModel | str](  # noqa: PLR6301
    self,
    model: ai.LoadedModel,
    system_prompt: str,
    user_prompt: str,
    output_format: type[T],
    call_seed: int,
    /,
    *,
    images: list[ai.AIImageInput] | None = None,
    tools: list[ai.AnyCallable] | None = None,
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
    if call_seed <= 1:  # for safety, but should never happen
      raise Error('call_seed must be a positive integer')
    config = lmstudio.LlmPredictionConfigDict(
      # Number of tokens to predict at most.
      # If set to false, the model will predict as many tokens as it wants.
      maxTokens=model.config['context'],
      # The temperature parameter for the prediction model. A higher value makes the predictions
      # more random, while a lower value makes the predictions more deterministic.
      # The value should be between 0 and 1.
      temperature=model.config['temperature'],
    )
    chat = lmstudio.Chat(system_prompt)
    chat.add_user_message(
      user_prompt,
      images=[lmstudio.prepare_image(b) for b in images] if images else None,  # type: ignore[arg-type]
    )
    # call the model
    logging.debug(f'Calling AI with config {config!r} and chat {chat!r}')
    llm: lmstudio.LLM = cast('lmstudio.LLM', model.model)
    try:
      if tools:
        # this is a tool-using call
        if output_format is not str:
          raise Error('Tools and return non-str output: unsupported')
        return _CallLMSAct(llm, chat, config, tools)  # type: ignore[return-value]
      # this is a normal call without tools: call, parse and return the result
      result: lmstudio.PredictionResult = _CallLMSRespond(llm, chat, config, output_format)
      return result.content if output_format is str else output_format.model_validate(result.parsed)  # type: ignore[return-value,attr-defined]
    except lmstudio.LMStudioServerError as err:
      raise Error(f'Error calling model {model.model_id!r}: {err}') from err


def _CallLMSRespond(
  llm: lmstudio.LLM,
  chat: lmstudio.Chat,
  config: lmstudio.LlmPredictionConfigDict,
  output_format: type,
) -> lmstudio.PredictionResult:
  result: lmstudio.PredictionResult = llm.respond(
    chat,
    config=config,
    response_format=None if output_format is str else output_format,
  )
  # log and check results
  logging.debug('Predicted tokens: %d', result.stats.predicted_tokens_count)
  logging.debug('Time to first token (seconds): %f', result.stats.time_to_first_token_sec)
  if result.stats.stop_reason != 'eosFound':
    raise Error(f'Unexpected stop reason {result.stats.stop_reason!r} while calling LMS')
  return result


def _CallLMSAct(
  llm: lmstudio.LLM,
  chat: lmstudio.Chat,
  config: lmstudio.LlmPredictionConfigDict,
  tools: list[ai.AnyCallable],
) -> str:
  """Call the LMStudio LLM using the Act API to support tool use.

  Args:
    llm: the LMStudio LLM instance to call
    chat: the LMStudio Chat instance containing the system and user messages
    config: the LMStudio LlmPredictionConfigDict with the prediction configuration
    tools: the list of tools (methods) to use during the call

  Returns:
    the string output from the model after processing the chat and tool calls

  """
  messages: list[str] = []
  hanging_calls: dict[str, str] = {}

  def _Act(message: lmstudio.AssistantResponse | lmstudio.ToolResultMessage) -> None:
    method_call: str
    for content in message.content:
      if isinstance(content, lmstudio.TextData):
        logging.debug(f'Model returned text content: {content.text!r}')
        # remove <think>...</think> part, if present
        all_content: str = ai.RE_THINK.sub('', content.text).strip()
        if all_content:
          messages.append(all_content)
      elif isinstance(content, lmstudio.FileHandle):
        logging.error(f'Model returned unexpected file handle: {content!r}')
      elif isinstance(content, lmstudio.ToolCallResultData):
        if not content.tool_call_id:
          raise Error(f'Tool response missing id: {content!r}')
        if content.tool_call_id not in hanging_calls:
          raise Error(f'Tool response with unknown id: {content!r}')
        method_call = hanging_calls.pop(content.tool_call_id)
        logging.info(f'{method_call} -> {content.content} (# {content.tool_call_id})')
      else:
        # this has to be a ToolCallRequestData
        if not content.tool_call_request.id:
          raise Error(f'Tool call missing id: {content!r}')
        args: str = (
          ', '.join(f'{f}={v!r}' for f, v in content.tool_call_request.arguments.items())
          if content.tool_call_request.arguments
          else ''
        )
        method_call = f'{content.tool_call_request.name}({args})'
        hanging_calls[content.tool_call_request.id] = method_call
        logging.debug(f'{method_call} -> {content.tool_call_request.id}')

  act_result: lmstudio.ActResult = llm.act(chat, config=config, tools=tools, on_message=_Act)
  logging.debug(f'Predicted tool calls: {act_result.rounds}')
  return '\n'.join(messages)  # type: ignore[return-value]


def _ExtractModelInfo(model: lmstudio.LLM, config: ai.AIModelConfig, /) -> ai.LoadedModel:
  """Extract and validate model information from the given LMStudio LLM instance.

  Args:
    model: the LMStudio LLM instance to extract information from
    config: the original AIModelConfig used for loading, for reference and validation

  Returns:
    a tuple of (AIModelConfig, ModelMetadata, LLM) with the extracted and validated model info

  Raises:
    Error: if the model information is invalid or does not meet the requirements of config

  """
  # do basic model validation and return if OK
  model_info = model.get_info()
  if not isinstance(model_info, lmstudio.LlmInstanceInfo):
    raise Error(f'Model {config["model_id"]} not a valid LLM instance')
  model_key: str = config['model_id']
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
      'vision': model_info.vision,
      'tooling': model_info.trained_for_tool_use,
      'context': n_ctx,
    }
  )
  if not new_config['seed'] or new_config['seed'] <= 1:  # for safety, but should never happen
    raise Error('Loaded LMStudio model config must have a seed')
  return ai.LoadedModel(
    model_id=model_key,
    seed_state=hashes.Hash256(base.IntToBytes(new_config['seed'])),
    config=new_config,
    metadata=cast('ai.AIModelMetadata', model_info.to_dict()),
    model=model,
  )
