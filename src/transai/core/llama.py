# SPDX-FileCopyrightText: Copyright 2026 <balparda@github.com> & <BellaKeri@github.com>
# SPDX-License-Identifier: Apache-2.0
"""llama.cpp AI library."""

from __future__ import annotations

import base64
import collections.abc
import contextlib
import gc
import json
import logging
import os
import pathlib
from typing import Any, cast

import llama_cpp
import pydantic
from llama_cpp import llama_chat_format, llama_speculative, llama_types
from lmstudio import json_api
from lmstudio._sdk_models import LlmToolFunctionDict
from transcrypto.core import hashes
from transcrypto.utils import base, saferandom

from transai.core import ai
from transai.utils import images as ai_images

_ToolID: collections.abc.Callable[[], str] = lambda: str(saferandom.RandInt(2**16, ai.AI_MAX_SEED))

_CLIP_KEYWORDS: set[str] = {
  'mmproj',
  'clip',
  'vision',
  'projector',
}
_TOOLING_KEYWORDS: set[str] = {
  'tool',
  'function',
  'functionary',
  'chatml-function-calling',
}
_REASONING_KEYWORDS: set[str] = {
  'reason',
  'thinking',
  'think',
  'deepseek-r1',
  'qwq',
  'qvq',
}

# Map of GGUF metadata hints to vision chat-handler classes; first match wins.
_LLAMA_VISION_HINTS: list[tuple[tuple[str, ...], type[llama_chat_format.Llava15ChatHandler]]] = [
  (
    ('qwen2.5-vl', 'qwen2_5_vl', 'qwen25vl', 'qwen2.5vl', 'qwen3-vl', 'qwen3_vl'),
    llama_chat_format.Qwen25VLChatHandler,
  ),
  (
    ('minicpm-v', 'minicpmv'),
    llama_chat_format.MiniCPMv26ChatHandler,
  ),
  (
    ('llama3-vision', 'llama-3-vision', 'llama3vision'),
    llama_chat_format.Llama3VisionAlphaChatHandler,
  ),
  (('moondream',), llama_chat_format.MoondreamChatHandler),
  (
    ('nanollava', 'nano-llava'),
    llama_chat_format.NanoLlavaChatHandler,
  ),
  (('obsidian',), llama_chat_format.ObsidianChatHandler),
  (
    ('llava-1.6', 'llava-v1.6', 'llava16'),
    llama_chat_format.Llava16ChatHandler,
  ),
  # generic llava / fallback — must stay last
  (('llava',), llama_chat_format.Llava15ChatHandler),
]


class Error(ai.Error):
  """Llama.cpp AI library error."""


class LlamaWorker(ai.AIWorker):
  """AI worker implementation using llama.cpp (llama-cpp-python). Use as context manager."""

  def __init__(
    self,
    models_root: pathlib.Path,
    *,
    timeout: float | None = ai.DEFAULT_TIMEOUT,
    verbose: bool = False,
  ) -> None:
    """Initialize the llama.cpp worker.

    Args:
      models_root: path to the directory containing the local GGUF models
      timeout (default=ai.DEFAULT_TIMEOUT): optional timeout in seconds for model loading and calls;
          if not given, defaults to ai.DEFAULT_TIMEOUT; can be set to None for no timeout
      verbose: whether to enable verbose logging

    Raises:
      Error: if the models_root is not a valid directory

    """
    super().__init__(timeout=timeout)
    self._models_root: pathlib.Path = models_root
    self._verbose: bool = verbose
    if not self._models_root:
      raise Error('models_root is required')
    self._models_root = self._models_root.expanduser().resolve()
    if not self._models_root.is_dir():
      raise Error(f'models_root is not a directory: {self._models_root}')
    logging.info(f'LLAMA @ {self._models_root}')

  def Close(self) -> None:
    """Close any started sessions, explicitly freeing vision handlers first.

    Overrides the parent Close() to additionally close vision (CLIP/mtmd) chat handlers
    before closing the underlying llama.cpp models. This ensures vision Metal residency
    sets are released BEFORE ggml_metal_device C++ static destructors run at process exit.

    In llama_cpp_python >= 0.3, Llava15ChatHandler (and subclasses like Qwen25VLChatHandler)
    use an ExitStack to register a mtmd_free() cleanup callback — they have NO __del__.
    This means setting chat_handler = None only orphans the handler; the ExitStack callbacks
    are never invoked, so mtmd_ctx/CLIP Metal residency sets remain registered forever,
    causing GGML_ASSERT([rsets->data count] == 0) in ggml_metal_device_free at process exit
    (triggered by click/typer's SystemExit → C exit() → __cxa_finalize_ranges).
    Explicitly calling _exit_stack.close() drains mtmd_free() synchronously, releasing
    the Metal context BEFORE llama_free() reduces the global device refcount to zero.

    """
    # drain each vision handler's ExitStack BEFORE closing the main llama.cpp model;
    # this invokes the mtmd_free() callback registered during _init_mtmd_context(),
    # properly tearing down the Metal context while the global device is still alive
    for loaded in self._loaded_models.values():
      if not isinstance(loaded.model, llama_cpp.Llama):
        continue
      handler = loaded.model.chat_handler
      if handler is None:
        continue
      logging.info(f'Releasing vision handler for {loaded.model_id!r}')
      if hasattr(handler, '_exit_stack'):
        logging.info(f'  Draining ExitStack (mtmd_free) for vision handler {loaded.model_id!r}')
        exit_stack = cast('contextlib.ExitStack', handler._exit_stack)  # type: ignore[attr-defined]  # noqa: SLF001
        try:
          exit_stack.close()  # invokes mtmd_free() → mtmd_cpp.mtmd_free(ctx) → Metal free
        except Exception as err:  # noqa: BLE001 --- external library callback, can raise anything
          logging.warning(f'  ExitStack close failed for {loaded.model_id!r}: {err}')
      loaded.model.chat_handler = None  # drop the Python reference now that resources are freed
    gc.collect()  # drain any remaining cyclic garbage
    super().Close()

  def _LoadNew(self, config: ai.AIModelConfig) -> ai.LoadedModel:
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
    # resolve GGUF files we can find, prioritizing explicit paths in the config
    gguf_path: pathlib.Path
    clip_path: pathlib.Path | None
    if config['model_path']:
      gguf_path = config['model_path'].expanduser().resolve()
      clip_path = (
        None if config['clip_path'] is None else config['clip_path'].expanduser().resolve()
      )
    else:
      gguf_path, clip_path = _FindGGUF(
        config['model_id'], self._FindModelDirectory(config['model_id'])
      )
    # setup vision handler
    handler: llama_chat_format.Llava15ChatHandler | None = None
    is_vision: bool = False
    if clip_path is not None:
      # detect the right handler from the model filename / id
      name_hint: str = gguf_path.stem.lower() + ' ' + config['model_id']
      handler_cls = _DetectVisionHandler(name_hint)
      if handler_cls is None:
        raise Error(
          f'Clip file {clip_path} found but no vision handler '
          f'detected for model {config["model_id"]!r}',
        )
      logging.info(f'Vision handler: {handler_cls.__name__!r} with clip {clip_path}')
      handler = handler_cls(clip_model_path=str(clip_path))
      is_vision = True
    if config['vision'] and not is_vision:
      raise Error(f'Vision requested but not loaded for model {config["model_id"]!r}')
    # speculative decoding
    draft_model: llama_speculative.LlamaPromptLookupDecoding | None = None
    spec_tokens: int = 0
    if config['spec_tokens'] is not None and (spec_tokens := config['spec_tokens']) > 0:
      draft_model = llama_speculative.LlamaPromptLookupDecoding(num_pred_tokens=spec_tokens)
    # load model
    logging.info(f'Loading {gguf_path}' + (f' + {spec_tokens} tokens' if draft_model else ''))  # noqa: G003
    with _SuppressNativeOutput(not self._verbose):
      llm: llama_cpp.Llama = llama_cpp.Llama(
        model_path=str(gguf_path),
        # my tests show this key does not matter if you set the query key
        seed=config['seed'] or -1,  # should never be None here, but for safety
        n_ctx=config['context'],
        temperature=config['temperature'],
        n_gpu_layers=config['gpu_layers'],
        use_mmap=config['use_mmap'],
        offload_kqv=True,
        flash_attn=config['flash'],
        type_k=config['kv_cache'],  # for now both share the same type...
        type_v=config['kv_cache'],
        chat_handler=handler,
        draft_model=draft_model,
        verbose=self._verbose,
      )
    # extract metadata from the loaded model
    metadata: ai.AIModelMetadata = dict(llm.metadata) if llm.metadata else {}
    md_text: str = _MetadataText(metadata)
    has_tooling: bool = any(x in md_text for x in _TOOLING_KEYWORDS)
    has_reasoning: bool = any(x in md_text for x in _REASONING_KEYWORDS)
    # finalize config and metadata, store and return
    new_config: ai.AIModelConfig = config.copy()
    new_config.update({
      'model_path': gguf_path,
      'clip_path': clip_path,
      'vision': is_vision,
      'tooling': has_tooling,
      'reasoning': has_reasoning,
    })
    if not new_config['seed'] or new_config['seed'] <= 1:  # for safety, but should never happen
      raise Error('Loaded llama.cpp model config must have a seed')
    return ai.LoadedModel(
      model_id=new_config['model_id'],
      seed_state=hashes.Hash256(base.IntToBytes(new_config['seed'])),
      config=new_config,
      metadata=metadata,
      model=llm,
    )

  def _Call[T: pydantic.BaseModel | str](  # noqa: C901
    self,
    model: ai.LoadedModel,
    system_prompt: str,
    user_prompt: str,
    output_format: type[T],
    call_seed: int,
    *,
    images: list[ai.AIImageInput] | None = None,
    tools: list[ai.AnyCallable] | None = None,
    chat_history: base.JSONDict | None = None,
  ) -> tuple[T, base.JSONDict]:
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
      chat_history (default=None): optional chat history to provide as context for the call;
          should be a JSON dict with the same format as the one returned by this method;
          if not given both `system_prompt` and `user_prompt` will be used as the initial messages;
          BEWARE, if given, `system_prompt` will be ignored, but `user_prompt` and `images` will
          be added to the chat before calling the model; MUTABLE!

    Returns:
      (T, base.JSONDict): a tuple of (
          the model output, either as a raw string or parsed into the given `output_format` class,
          a JSON dict with the chat history, INCLUDING the response so conversation can continue)

    Raises:
      Error: if the model does not support the given inputs, or if there is any error calling

    """
    if call_seed <= 1:  # for safety, but should never happen
      raise Error('call_seed must be a positive integer')
    # build messages, start with system prompt
    messages: list[base.JSONDict] = (
      cast('list[base.JSONDict]', chat_history.get('messages', []))
      if chat_history
      else [{'role': 'system', 'content': system_prompt}]
    )
    if images:
      # vision request: multi-modal user message
      if not model.config['vision']:
        raise Error(f'Model {model.model_id!r} does not support vision but images were provided')
      # add the text part of the user message
      parts: list[base.JSONDict] = [{'type': 'text', 'text': user_prompt}]
      # for llama.cpp, we need to convert images to data URIs and include them in the message
      for img in images:
        # down-scale large images to stay within the KV-cache budget
        img_bytes: bytes = ai_images.ResizeImageForVision(  # convert do 1024px max PNG single-frame
          pathlib.Path(img).expanduser().resolve().read_bytes()
          if isinstance(img, (str, pathlib.Path))
          else img
        )
        parts.append({
          'type': 'image_url',
          'image_url': {'url': _ImageToDataURI(img_bytes, 'image/png')},
        })
      # add the parts (text + images) as a single user message with mixed content
      messages.append({'role': 'user', 'content': parts})
    else:
      # text-only user message, so add the rest
      messages.append({'role': 'user', 'content': user_prompt})
    # call the model
    try:
      result: llama_types.CreateChatCompletionResponse
      llm: llama_cpp.Llama = cast('llama_cpp.Llama', model.model)
      if tools:
        # tool-use call: requires tooling-capable model and str output
        if not model.config['tooling']:
          raise Error(f'Model {model.model_id!r} does not support tools but tools were provided')
        if output_format is not str:
          raise Error('Tool-use calls require str output_format; structured output is unsupported')
        tool_map: dict[str, ai.AnyCallable] = {func.__name__: func for func in tools}
        llm_tools = json_api.ChatResponseEndpoint.parse_tools(tools)[0].tools
        if not llm_tools:
          raise Error(
            'No valid tools found; make sure the functions are properly typed and have docstrings'
          )
        # tools call: messages is extended in-place with full conversation history
        act_output: str = _CallLlamaAct(
          llm,
          messages,
          [t.to_dict() for t in llm_tools],
          tool_map,
          model.config,
          call_seed,
          self._verbose,
        )
        messages.append({'role': 'assistant', 'content': act_output})
        return (act_output, chat_history or {'messages': messages})  # type: ignore[return-value]
      # non-tool call
      schema: base.JSONDict | None = None
      if output_format is not str:
        schema = output_format.model_json_schema()  # type: ignore[attr-defined]
        schema.pop('$defs', None)  # pyright: ignore[reportUnknownMemberType]
        schema.pop('title', None)  # pyright: ignore[reportUnknownMemberType]
      logging.debug(
        f'Calling AI {output_format}/{call_seed} config {model.config!r}:\n{messages!r}'
      )
      with _SuppressNativeOutput(not self._verbose):
        result = llm.create_chat_completion(  # type: ignore[assignment]
          messages=messages,  # type: ignore[arg-type]
          response_format={'type': 'text'}
          if output_format is str
          else {'type': 'json_object', 'schema': schema},
          max_tokens=model.config['context'],
          temperature=model.config['temperature'],
          seed=call_seed,  # my tests have shown this is the only seed that matters
        )
      # return content according to the output format
      content: str = _ExtractContent(result)
      if output_format is str:
        # content is already a string
        messages.append({'role': 'assistant', 'content': content})
        return (content, chat_history or {'messages': messages})  # type: ignore[attr-defined,return-value]
      # content is a JSON object in string form, so parse it and validate with the pydantic model
      json_content: base.JSONDict = cast('base.JSONDict', json.loads(content))
      messages.append({'role': 'assistant', 'content': json_content})
      return (output_format.model_validate(json_content), chat_history or {'messages': messages})  # type: ignore[attr-defined,return-value]
    except (ValueError, RuntimeError) as err:
      raise Error(f'Error calling model {model.model_id!r}: {err}') from err

  def _FindModelDirectory(self, model_id: str) -> pathlib.Path:
    """Recursively search for exactly one directory matching *model_id*.

    The comparison is case-insensitive on the directory name.

    Args:
      model_id: the model identifier to match against directory names

    Returns:
      the single matching directory path

    Raises:
      Error: if zero or more than one directory matches

    """
    target: str = model_id.lower().rsplit('@', 1)[0].split('/', 1)[-1]  # ignore prefix/suffix hints
    matches: list[pathlib.Path] = [
      p for p in self._models_root.rglob('*') if p.is_dir() and target in p.name.lower()
    ]
    if not matches:
      raise Error(f'No directory matching model_id {target!r} found under {self._models_root}')
    if len(matches) > 1:
      raise Error(f'Ambiguous: {len(matches)} directories match model_id {target!r}: {matches!r}')
    return matches[0]


@contextlib.contextmanager
def _SuppressNativeOutput(suppress: bool) -> collections.abc.Iterator[None]:
  """Redirect C-level stdout/stderr to devnull (e.g. llama.cpp grammar logs)."""
  if not suppress:
    yield
    return
  devnull: int = os.open(os.devnull, os.O_WRONLY)
  old_stdout: int = os.dup(1)
  old_stderr: int = os.dup(2)
  try:
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    yield
  finally:
    os.dup2(old_stdout, 1)
    os.dup2(old_stderr, 2)
    os.close(old_stdout)
    os.close(old_stderr)
    os.close(devnull)


def _FindGGUF(model_id: str, model_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path | None]:
  """Find the primary GGUF model file in *model_dir* and the CLIP GGUF, if any.

  Picks the largest file (handles typical single-file models and picks the biggest
  shard when multiple files are present)...

  Args:
    model_id: the model identifier; we will use the '@' suffix if present to prioritize GGUF files
    model_dir: directory expected to contain GGUF file(s)

  Returns:
    (main GGUF path, clip GGUF path or None)

  Raises:
    Error: if no suitable GGUF files are found

  """
  gguf_candidates: set[pathlib.Path] = {
    f for f in model_dir.iterdir() if f.is_file() and f.suffix.lower() == '.gguf'
  }
  clip_candidates: set[pathlib.Path] = {
    g for g in gguf_candidates if any(kw in g.name.lower() for kw in _CLIP_KEYWORDS)
  }
  model_candidates: set[pathlib.Path] = {
    g for g in gguf_candidates if not any(kw in g.name.lower() for kw in _CLIP_KEYWORDS)
  }
  if not model_candidates:
    raise Error(f'No GGUF model files found in {model_dir}')
  if len(model_candidates) > 1 and '@' in model_id:
    target: str = model_id.lower().rsplit('@', 1)[-1]
    better_candidates: set[pathlib.Path] = {m for m in model_candidates if target in m.name.lower()}
    model_candidates = better_candidates or model_candidates
  return (
    max(model_candidates, key=lambda p: p.stat().st_size),
    max(clip_candidates, key=lambda p: p.stat().st_size) if clip_candidates else None,
  )


def _DetectVisionHandler(metadata_text: str) -> type[llama_chat_format.Llava15ChatHandler] | None:
  """Choose the best vision chat-handler from GGUF metadata.

  Args:
    metadata_text: lower-cased searchable metadata string

  Returns:
    handler class, or ``None`` if nothing matched

  """
  for hints, handler_cls in _LLAMA_VISION_HINTS:
    if any(h in metadata_text for h in hints):
      return handler_cls
  return None


def _MetadataText(metadata: ai.AIModelMetadata) -> str:
  """Concatenate all GGUF metadata into a single searchable string.

  Args:
    metadata: raw metadata dict from ``Llama.metadata``

  Returns:
    lower-cased string joining all keys and values

  """
  return ' '.join(f'{k}={v}' for k, v in metadata.items()).lower()


def _ImageToDataURI(image_bytes: bytes, mime: str = 'image/png') -> str:
  """Encode raw image bytes as a ``data:`` URI.

  Args:
    image_bytes: raw binary image data
    mime: MIME type (default ``image/png``)

  Returns:
    ``data:<mime>;base64,<encoded>`` string

  """
  b64: str = base64.b64encode(image_bytes).decode('ascii')
  return f'data:{mime};base64,{b64}'


def _CallLlamaAct(
  llm: llama_cpp.Llama,
  messages: list[base.JSONDict],
  tool_defs: list[LlmToolFunctionDict],
  tool_map: dict[str, ai.AnyCallable],
  config: ai.AIModelConfig,
  call_seed: int,
  verbose: bool = False,
) -> str:
  """Execute a tool-use loop with a llama.cpp model.

  Calls the model repeatedly, executing any requested tool calls and feeding the results
  back into the conversation, until the model produces a final text response.

  Args:
    llm: the llama.cpp Llama instance to call
    messages: initial message list (system + user); extended in-place with assistant and
        tool result messages as the conversation progresses
    tool_defs: OpenAI-format tool definition dicts
    tool_map: mapping from tool function name to its callable
    config: AIModelConfig with parameters for the call
    call_seed: the pre-computed seed to use for this call, derived from the model's seed state
    verbose: whether to enable verbose logging for the tool-use loop

  Returns:
    concatenated text content produced by the model across all rounds, joined by newlines

  """
  accumulated: list[str] = []
  result: llama_types.CreateChatCompletionResponse
  while True:
    logging.debug(f'Calling AI TOOL/{call_seed} config {config!r}/{tool_defs!r}:\n{messages!r}')
    with _SuppressNativeOutput(not verbose):
      result = llm.create_chat_completion(  # type: ignore[assignment]
        messages=messages,  # type: ignore[arg-type]
        tools=tool_defs,  # type: ignore[arg-type]
        tool_choice='auto',
        response_format={'type': 'text'},
        max_tokens=config['context'],
        temperature=config['temperature'],
        seed=call_seed,
      )
    # parse the model response
    content, tool_calls = _DetectToolHandler(config['model_id'])(_ExtractContent(result))  # type: ignore[arg-type]
    # record the assistant message (with tool_calls) in the conversation history
    if content:
      accumulated.append(content)
    if not tool_calls:
      break  # no more tool calls, we're done!
    messages.append({
      'role': 'assistant',
      'content': content or '',
      'tool_calls': tool_calls,
    })
    _ExecuteToolCalls(tool_calls, tool_map, messages)  # execute calls, append results
  # ended back-and-forth between model and tools; return accumulated content
  return '\n'.join(accumulated)


def _QwenDecode(content: str) -> tuple[str | None, list[base.JSONDict]]:
  tool_parts: list[str] = [part.strip() for part in ai.RE_TOOL_CALL.findall(content)]
  all_content = ai.RE_TOOL_CALL.sub('', content).strip()
  return (all_content or None, [{'id': _ToolID(), 'function': json.loads(t)} for t in tool_parts])


# Map of GGUF metadata hints to tool chat-handler classes; first match wins.
_MODEL_TOOL_DECODERS: list[
  tuple[tuple[str, ...], collections.abc.Callable[[str], tuple[str | None, list[base.JSONDict]]]]
] = [
  (('qwen2', 'qwen3'), _QwenDecode),
]


def _DetectToolHandler(
  metadata_text: str,
) -> collections.abc.Callable[[str], tuple[str | None, list[base.JSONDict]]]:
  """Choose the best tool tool-handler from GGUF metadata.

  Args:
    metadata_text: lower-cased searchable metadata string

  Returns:
    handler class, or ``None`` if nothing matched

  Raises:
    Error: not found

  """
  for hints, handler_cls in _MODEL_TOOL_DECODERS:
    if any(h in metadata_text for h in hints):
      return handler_cls
  raise Error(f'Model metadata does not match any known tool-handling patterns: {metadata_text!r}')


def _ExecuteToolCalls(
  tool_calls: list[base.JSONDict],
  tool_map: dict[str, ai.AnyCallable],
  messages: list[base.JSONDict],
) -> None:
  """Execute a list of tool calls and append their results to the message history.

  Args:
    tool_calls: list of tool call dicts from the model response
    tool_map: mapping from tool function name to its callable
    messages: conversation message list; each tool result is appended in-place

  Raises:
    Error: if the model calls an unknown tool, the arguments are invalid JSON,
        or the tool callable raises an exception

  """  # noqa: DOC501
  # loop over the tool calls in the order given by the model, execute them, and append results
  for tc in tool_calls:
    # get tool name and arguments
    call_id: str = tc.get('id', '')  # type: ignore[assignment]
    func_name: str = tc.get('function', {}).get('name', '')  # type: ignore[assignment,union-attr]
    args_str: str | base.JSONDict = tc.get('function', {}).get('arguments') or {}  # type: ignore[assignment,union-attr]
    if func_name not in tool_map:
      raise Error(f'Model called unknown tool {func_name!r}; available: {sorted(tool_map)!r}')
    # parse arguments (should be a JSON string or dict, depending on the model/handler)
    try:
      args: base.JSONDict = json.loads(args_str) if isinstance(args_str, str) else args_str
    except json.JSONDecodeError as err:
      raise Error(f'Tool {func_name!r} received invalid JSON arguments: {args_str!r}') from err
    # we should be good to execute now; log the call and arguments for debugging
    args_repr: str = ', '.join(f'{k}={v!r}' for k, v in args.items())
    logging.debug(f'Calling tool {func_name}({args_repr}) -> {call_id}')
    tool_result: Any
    try:
      try:
        tool_result = tool_map[func_name](**args)
      except TypeError as err:
        # try to fall back to positional args for callables that reject keyword args (e.g. builtins)
        if 'keyword arguments' not in str(err):
          raise  # not a kwarg issue, re-raise
        logging.debug(f'Tool {func_name!r} rejected keyword args, retrying positionally')
        tool_result = tool_map[func_name](*args.values())
    except Exception as err:  # noqa: BLE001 --- we are purposeful in catching all exceptions
      logging.error(f'Error: {func_name!r} raised: {err}')
      tool_result = err  # we will feed the exception info back to the model as the tool result
    logging.info(f'Tool {func_name}({args_repr}) -> {tool_result!r} (# {call_id})')
    messages.append({
      'role': 'tool',
      'tool_call_id': call_id,
      'name': func_name,
      'content': repr(tool_result),
    })


def _ExtractContent(result: llama_types.CreateChatCompletionResponse) -> str:
  """Pull text content from a chat completion response.

  Args:
    result: raw response from LLM

  Returns:
    the text content string

  Raises:
    Error: if the response contains no choices or empty content

  """
  # get 1st choice
  choices: list[Any] = result.get('choices', [])
  if not choices:
    raise Error('Model returned no choices')
  # check that reason is 'stop' (finished normally)
  if (finish_reason := choices[0].get('finish_reason', '')) != 'stop':
    raise Error(f'Model returned finish_reason={finish_reason!r}, expected "stop"')
  # extract content
  content: str = choices[0].get('message', {}).get('content', '').strip()
  if not content:
    raise Error(f'Model returned empty content (finish_reason={finish_reason!r})')
  # log
  logging.debug(f'Model response content:\n{content!r}')
  if usage := result.get('usage'):
    logging.debug(
      f'Tokens: prompt={usage.get("prompt_tokens", 0)}, '
      f'completion={usage.get("completion_tokens", 0)}, '
      f'total={usage.get("total_tokens", 0)}',
    )
  # now that we logged we clean <think> content, if any
  return ai.RE_THINK.sub('', content).strip()
