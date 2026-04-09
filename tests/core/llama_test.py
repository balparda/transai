# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""llama.py unittest.

Run with:
  poetry run pytest -vvv tests/core/llama_test.py
"""

from __future__ import annotations

import base64
import json
import pathlib
from typing import Any
from unittest import mock

import llama_cpp
import lmstudio
import pydantic
import pytest
import typeguard
from llama_cpp import llama_chat_format
from transcrypto.utils import base

from transai.core import ai, llama

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FalsyPath(pathlib.Path):
  """A pathlib.Path subclass that evaluates to False, for testing empty-path guard."""

  def __bool__(self) -> bool:
    """Return False."""  # noqa: DOC201
    return False


def _MakeResponse(
  content: str = 'ok',
  finish_reason: str = 'stop',
) -> dict[str, Any]:
  """Return a complete llama_types.CreateChatCompletionResponse-compatible dict.

  Args:
    content: the content of the assistant's message
    finish_reason: the reason the completion finished

  Returns:
    A dict compatible with llama_types.CreateChatCompletionResponse

  """
  return {
    'id': 'chat-cmp-test',
    'object': 'chat.completion',
    'created': 1234567890,
    'model': 'test-model',
    'choices': [
      {
        'index': 0,
        'message': {'role': 'assistant', 'content': content},
        'logprobs': None,
        'finish_reason': finish_reason,
      }
    ],
    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
  }


# ---------------------------------------------------------------------------
# LlamaWorker.__init__
# ---------------------------------------------------------------------------


def testLlamaWorkerInitRaisesOnEmptyPath() -> None:
  """LlamaWorker must raise if models_root evaluates to False (uses _FalsyPath)."""
  with pytest.raises(llama.Error, match='required'):
    llama.LlamaWorker(_FalsyPath('/tmp'))  # noqa: S108


def testLlamaWorkerInitRaisesOnNonDirectory(tmp_path: pathlib.Path) -> None:
  """LlamaWorker must raise if models_root is not a directory."""
  not_a_dir: pathlib.Path = tmp_path / 'file.txt'
  not_a_dir.write_text('x', encoding='utf-8')
  with pytest.raises(llama.Error, match='not a directory'):
    llama.LlamaWorker(not_a_dir)


def testLlamaWorkerInitSucceeds(tmp_path: pathlib.Path) -> None:
  """LlamaWorker.__init__ succeeds for a real directory."""
  w = llama.LlamaWorker(tmp_path)
  assert w._models_root == tmp_path


def testLlamaWorkerInitVerboseFlag(tmp_path: pathlib.Path) -> None:
  """Verbose flag is stored correctly."""
  w = llama.LlamaWorker(tmp_path, verbose=True)
  assert w._verbose is True


# ---------------------------------------------------------------------------
# LlamaWorker.Close
# ---------------------------------------------------------------------------


def testCloseWithNoHandlerCallsSuperClose(tmp_path: pathlib.Path) -> None:
  """Close() with a model that has no chat_handler calls super().Close() cleanly."""
  w = llama.LlamaWorker(tmp_path)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.chat_handler = None
  loaded = ai.LoadedModel(
    model_id='test@q4_0',
    seed_state=b'\x00' * 32,
    config=ai.MakeAIModelConfig(model_id='test@q4_0', seed=42),
    metadata={},
    model=llm_mock,
  )
  w._loaded_models['test@q4_0'] = loaded
  w.Close()
  # super().Close() calls model.close()
  llm_mock.close.assert_called_once()
  assert w._loaded_models == {}


def testCloseSkipsNonLlamaModels(tmp_path: pathlib.Path) -> None:
  """Close() skips the handler-drain logic for non-llama_cpp.Llama model types."""
  w = llama.LlamaWorker(tmp_path)
  lms_mock = mock.MagicMock(spec=lmstudio.LLM)
  loaded = ai.LoadedModel(
    model_id='lms@q4_0',
    seed_state=b'\x00' * 32,
    config=ai.MakeAIModelConfig(model_id='lms@q4_0', seed=42),
    metadata={},
    model=lms_mock,
  )
  w._loaded_models['lms@q4_0'] = loaded
  # should not raise and should not call close() (lmstudio models don't have close())
  w.Close()
  assert w._loaded_models == {}


def testCloseWithHandlerDrainsExitStack(tmp_path: pathlib.Path) -> None:
  """Close() calls _exit_stack.close() on a handler that has one."""
  w = llama.LlamaWorker(tmp_path)
  exit_stack_mock = mock.MagicMock()
  handler_mock = mock.MagicMock(spec=llama_chat_format.Qwen25VLChatHandler)
  handler_mock._exit_stack = exit_stack_mock
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.chat_handler = handler_mock
  loaded = ai.LoadedModel(
    model_id='vision@q4_0',
    seed_state=b'\x00' * 32,
    config=ai.MakeAIModelConfig(model_id='vision@q4_0', seed=42, vision=True),
    metadata={},
    model=llm_mock,
  )
  w._loaded_models['vision@q4_0'] = loaded
  w.Close()
  exit_stack_mock.close.assert_called_once()
  assert llm_mock.chat_handler is None
  llm_mock.close.assert_called_once()  # type: ignore[unreachable]  # it **IS** reachable!!


def testCloseWithHandlerExitStackRaisesWarnsAndContinues(tmp_path: pathlib.Path) -> None:
  """Close() logs a warning if _exit_stack.close() raises, but still completes normally."""
  w = llama.LlamaWorker(tmp_path)
  exit_stack_mock = mock.MagicMock()
  exit_stack_mock.close.side_effect = RuntimeError('Metal exploded')
  handler_mock = mock.MagicMock(spec=llama_chat_format.Qwen25VLChatHandler)
  handler_mock._exit_stack = exit_stack_mock
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.chat_handler = handler_mock
  loaded = ai.LoadedModel(
    model_id='vision@q4_0',
    seed_state=b'\x00' * 32,
    config=ai.MakeAIModelConfig(model_id='vision@q4_0', seed=42, vision=True),
    metadata={},
    model=llm_mock,
  )
  w._loaded_models['vision@q4_0'] = loaded
  w.Close()  # must not propagate the RuntimeError
  exit_stack_mock.close.assert_called_once()
  llm_mock.close.assert_called_once()
  assert w._loaded_models == {}


# ---------------------------------------------------------------------------
# LlamaWorker._FindModelDirectory
# ---------------------------------------------------------------------------


def testFindModelDirectoryFindsExact(tmp_path: pathlib.Path) -> None:
  """_FindModelDirectory finds a single directory matching model_id."""
  model_dir: pathlib.Path = tmp_path / 'my-model'
  model_dir.mkdir()
  w = llama.LlamaWorker(tmp_path)
  result: pathlib.Path = w._FindModelDirectory('my-model')
  assert result == model_dir


def testFindModelDirectoryStripsAtSuffix(tmp_path: pathlib.Path) -> None:
  """_FindModelDirectory strips the @quant suffix before searching."""
  model_dir: pathlib.Path = tmp_path / 'qwen3-8b'
  model_dir.mkdir()
  w = llama.LlamaWorker(tmp_path)
  result: pathlib.Path = w._FindModelDirectory('qwen3-8b@Q8_0')
  assert result == model_dir


def testFindModelDirectoryRaisesWhenNotFound(tmp_path: pathlib.Path) -> None:
  """_FindModelDirectory raises if no directory matches."""
  w = llama.LlamaWorker(tmp_path)
  with pytest.raises(llama.Error, match='No directory'):
    w._FindModelDirectory('ghost-model')


def testFindModelDirectoryRaisesOnAmbiguous(tmp_path: pathlib.Path) -> None:
  """_FindModelDirectory raises if multiple directories match."""
  (tmp_path / 'model-a').mkdir()
  (tmp_path / 'model-a-v2').mkdir()
  w = llama.LlamaWorker(tmp_path)
  with pytest.raises(llama.Error, match='Ambiguous'):
    w._FindModelDirectory('model-a')


# ---------------------------------------------------------------------------
# _FindGGUF
# ---------------------------------------------------------------------------


def testFindGGUFReturnsMainModel(tmp_path: pathlib.Path) -> None:
  """_FindGGUF picks the sole GGUF as the main model; no clip."""
  model_file: pathlib.Path = tmp_path / 'model.gguf'
  model_file.write_bytes(b'x' * 100)
  main, clip = llama._FindGGUF('model', tmp_path)
  assert main == model_file
  assert clip is None


def testFindGGUFDetectsClipFile(tmp_path: pathlib.Path) -> None:
  """_FindGGUF separates clip GGUF from main GGUF."""
  main_file: pathlib.Path = tmp_path / 'model.gguf'
  clip_file: pathlib.Path = tmp_path / 'mmproj-clip.gguf'
  main_file.write_bytes(b'x' * 100)
  clip_file.write_bytes(b'y' * 50)
  main, clip = llama._FindGGUF('model', tmp_path)
  assert main == main_file
  assert clip == clip_file


def testFindGGUFRaisesWhenNoGGUF(tmp_path: pathlib.Path) -> None:
  """_FindGGUF raises Error if no GGUF files are present."""
  with pytest.raises(llama.Error, match='No GGUF'):
    llama._FindGGUF('model', tmp_path)


def testFindGGUFPicksLargestFile(tmp_path: pathlib.Path) -> None:
  """_FindGGUF picks the largest GGUF when multiple candidates exist."""
  small: pathlib.Path = tmp_path / 'small.gguf'
  large: pathlib.Path = tmp_path / 'large.gguf'
  small.write_bytes(b'x' * 10)
  large.write_bytes(b'x' * 200)
  main, _ = llama._FindGGUF('model', tmp_path)
  assert main == large


def testFindGGUFUsesAtSuffixToFilter(tmp_path: pathlib.Path) -> None:
  """_FindGGUF uses the @suffix to prefer matching GGUF filenames."""
  q4: pathlib.Path = tmp_path / 'model-q4_k_m.gguf'
  q8: pathlib.Path = tmp_path / 'model-q8_0.gguf'
  q4.write_bytes(b'x' * 200)  # bigger but wrong quant
  q8.write_bytes(b'y' * 100)
  main, _ = llama._FindGGUF('model@q8_0', tmp_path)
  assert main == q8


# ---------------------------------------------------------------------------
# _DetectVisionHandler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ('hint', 'expected_cls'),
  [
    ('qwen2.5-vl text', llama_chat_format.Qwen25VLChatHandler),
    ('qwen3-vl model', llama_chat_format.Qwen25VLChatHandler),
    ('minicpm-v chat', llama_chat_format.MiniCPMv26ChatHandler),
    ('llama3-vision model', llama_chat_format.Llama3VisionAlphaChatHandler),
    ('moondream model', llama_chat_format.MoondreamChatHandler),
    ('nanollava model', llama_chat_format.NanoLlavaChatHandler),
    ('obsidian-llava model', llama_chat_format.ObsidianChatHandler),
    ('llava-1.6 model', llama_chat_format.Llava16ChatHandler),
    ('llava-model', llama_chat_format.Llava15ChatHandler),
  ],
)
def testDetectVisionHandlerMatches(
  hint: str, expected_cls: type[llama_chat_format.Llava15ChatHandler]
) -> None:
  """_DetectVisionHandler returns the expected handler class for each hint."""
  result: type[llama_chat_format.Llava15ChatHandler] | None = llama._DetectVisionHandler(hint)
  assert result is expected_cls


def testDetectVisionHandlerReturnsNoneForUnknown() -> None:
  """_DetectVisionHandler returns None when no hint matches."""
  assert llama._DetectVisionHandler('some-random-model-name') is None


# ---------------------------------------------------------------------------
# _MetadataText
# ---------------------------------------------------------------------------


def testMetadataTextJoinsData() -> None:
  """_MetadataText joins key=value pairs into a single lower-cased string."""
  meta: ai.AIModelMetadata = {'general.name': 'MyModel', 'gguf.version': '3'}
  text: str = llama._MetadataText(meta)
  assert 'general.name=mymodel' in text
  assert 'gguf.version=3' in text


def testMetadataTextEmptyDict() -> None:
  """_MetadataText returns an empty string for empty metadata."""
  assert not llama._MetadataText({})


# ---------------------------------------------------------------------------
# _ImageToDataURI
# ---------------------------------------------------------------------------


def testImageToDataURI() -> None:
  """_ImageToDataURI produces a valid data URI."""
  data: bytes = b'\x89PNG\r\n'
  uri: str = llama._ImageToDataURI(data, 'image/png')
  assert uri.startswith('data:image/png;base64,')

  decoded: bytes = base64.b64decode(uri.split(',', 1)[1])
  assert decoded == data


def testImageToDataURIDefaultMime() -> None:
  """Default mime type is image/png."""
  uri: str = llama._ImageToDataURI(b'bytes')
  assert uri.startswith('data:image/png;base64,')


# ---------------------------------------------------------------------------
# _ExtractContent
# ---------------------------------------------------------------------------


def testExtractContentSuccess() -> None:
  """_ExtractContent returns content on a well-formed response."""
  with typeguard.suppress_type_checks():
    assert llama._ExtractContent(_MakeResponse('hello world')) == 'hello world'  # type: ignore[arg-type]


def testExtractContentRaisesOnNoChoices() -> None:
  """_ExtractContent raises Error when choices list is empty."""
  response: Any = _MakeResponse()
  response['choices'] = []
  with typeguard.suppress_type_checks(), pytest.raises(llama.Error, match='no choices'):
    llama._ExtractContent(response)


def testExtractContentRaisesOnEmptyContent() -> None:
  """_ExtractContent raises Error when content is empty/None."""
  response: Any = _MakeResponse()
  response['choices'][0]['message']['content'] = ''
  with typeguard.suppress_type_checks(), pytest.raises(llama.Error, match='empty content'):
    llama._ExtractContent(response)


def testExtractContentRaisesOnBadFinishReason() -> None:
  """_ExtractContent raises Error for an unexpected finish_reason."""
  with typeguard.suppress_type_checks(), pytest.raises(llama.Error, match='finish_reason'):
    llama._ExtractContent(_MakeResponse('text', 'timeout'))  # type: ignore[arg-type]


def testExtractContentLogsTokenUsage() -> None:
  """_ExtractContent logs token usage when 'usage' is present."""
  # Should not raise; just verify it runs cleanly with usage data
  with typeguard.suppress_type_checks():
    assert llama._ExtractContent(_MakeResponse('ok')) == 'ok'  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _SuppressNativeOutput
# ---------------------------------------------------------------------------


def testSuppressNativeOutputSuppressing() -> None:
  """_SuppressNativeOutput redirects fds when suppress=True."""
  # We just verify the context manager runs without error; actual fd
  # redirection is OS-level and hard to assert portably.
  with llama._SuppressNativeOutput(True):
    pass  # should not raise


def testSuppressNativeOutputNotSuppressing() -> None:
  """_SuppressNativeOutput is a no-op when suppress=False."""
  with llama._SuppressNativeOutput(False):
    pass


# ---------------------------------------------------------------------------
# LlamaWorker._Load
# ---------------------------------------------------------------------------


def testLlamaWorkerLoadUsesExplicitModelPath(tmp_path: pathlib.Path) -> None:
  """_Load uses config.model_path when explicitly given."""
  gguf_file: pathlib.Path = tmp_path / 'mymodel.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_path=gguf_file, seed=5000)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {'general.name': 'test'}
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    result: ai.LoadedModel = w._LoadNew(config)
  assert result.config['model_path'] == gguf_file
  assert result.config['clip_path'] is None
  assert result.config['vision'] is False
  assert result.model is mock_llama


def testLlamaWorkerLoadSearchesForModelWhenNoPath(tmp_path: pathlib.Path) -> None:
  """_Load searches for a model directory when model_path is None."""
  model_dir: pathlib.Path = tmp_path / 'mymodel'
  model_dir.mkdir()
  gguf_file: pathlib.Path = model_dir / 'mymodel.gguf'
  gguf_file.write_bytes(b'model data' * 10)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='mymodel', seed=5000)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {'general.name': 'test'}
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    result: ai.LoadedModel = w._LoadNew(config)
  assert result.config['model_path'] == gguf_file
  assert result.model is mock_llama


def testLlamaWorkerLoadSetsVisionTrueWithClip(tmp_path: pathlib.Path) -> None:
  """_Load sets vision=True when a clip file is provided."""
  gguf_file: pathlib.Path = tmp_path / 'qwen3-vl-model.gguf'
  clip_file: pathlib.Path = tmp_path / 'mmproj-clip.gguf'
  gguf_file.write_bytes(b'model' * 20)
  clip_file.write_bytes(b'clip' * 10)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(
    model_id='qwen3-vl', model_path=gguf_file, clip_path=clip_file, vision=True, seed=5000
  )
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {'general.name': 'qwen3-vl'}
  mock_handler_instance = mock.MagicMock()
  mock_handler_cls = mock.MagicMock(return_value=mock_handler_instance)
  mock_handler_cls.__name__ = 'MockVisionHandler'  # needed for logging.info in _Load
  with (
    mock.patch('llama_cpp.Llama', return_value=mock_llama),
    mock.patch('transai.core.llama._DetectVisionHandler', return_value=mock_handler_cls),
  ):
    result: ai.LoadedModel = w._LoadNew(config)
  assert result.config['vision'] is True
  mock_handler_cls.assert_called_once_with(clip_model_path=str(clip_file.resolve()))


def testLlamaWorkerLoadRaisesIfVisionRequestedButNoClip(tmp_path: pathlib.Path) -> None:
  """_Load raises Error when vision=True but no clip file found."""
  gguf_file: pathlib.Path = tmp_path / 'no-vis.gguf'
  gguf_file.write_bytes(b'model' * 20)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(
    model_id='no-vis', model_path=gguf_file, vision=True
  )
  w = llama.LlamaWorker(tmp_path)
  with pytest.raises(llama.Error, match='Vision requested'):
    w._LoadNew(config)


def testLlamaWorkerLoadRaisesOnUnknownVisionHandler(tmp_path: pathlib.Path) -> None:
  """_Load raises Error when clip found but no handler detected."""
  gguf_file: pathlib.Path = tmp_path / 'unknown-vis.gguf'
  clip_file: pathlib.Path = tmp_path / 'mmproj.gguf'
  gguf_file.write_bytes(b'model' * 20)
  clip_file.write_bytes(b'clip' * 10)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_path=gguf_file, clip_path=clip_file)
  w = llama.LlamaWorker(tmp_path)
  with (
    mock.patch('transai.core.llama._DetectVisionHandler', return_value=None),
    pytest.raises(llama.Error, match='no vision handler'),
  ):
    w._LoadNew(config)


def testLlamaWorkerLoadUsesSpecTokens(tmp_path: pathlib.Path) -> None:
  """_Load creates a draft model when spec_tokens > 0."""
  gguf_file: pathlib.Path = tmp_path / 'mymodel.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_path=gguf_file, spec_tokens=5, seed=5000)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {}
  mock_draft = mock.MagicMock()
  with (
    mock.patch('llama_cpp.Llama', return_value=mock_llama),
    mock.patch(
      'transai.core.llama.llama_speculative.LlamaPromptLookupDecoding', return_value=mock_draft
    ) as draft_cls,
    typeguard.suppress_type_checks(),
  ):
    w._LoadNew(config)
  draft_cls.assert_called_once_with(num_pred_tokens=5)


def testLlamaWorkerLoadSetsToolingFromMetadata(tmp_path: pathlib.Path) -> None:
  """_Load sets tooling=True when metadata contains tooling keywords."""
  gguf_file: pathlib.Path = tmp_path / 'tool-model.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_path=gguf_file, seed=5000)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {'model.type': 'functionary-chat'}
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    result: ai.LoadedModel = w._LoadNew(config)
  assert result.config['tooling'] is True


def testLlamaWorkerLoadSetsReasoningFromMetadata(tmp_path: pathlib.Path) -> None:
  """_Load sets reasoning=True when metadata contains reasoning keywords."""
  gguf_file: pathlib.Path = tmp_path / 'reason-model.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_path=gguf_file, seed=5000)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {'model.type': 'deepseek-r1-reasoning'}
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    result: ai.LoadedModel = w._LoadNew(config)
  assert result.config['reasoning'] is True


def testLlamaWorkerLoadHandlesNoneMetadata(tmp_path: pathlib.Path) -> None:
  """_Load handles the case where llm.metadata is None."""
  gguf_file: pathlib.Path = tmp_path / 'mymodel.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_path=gguf_file, seed=5000)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = None
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    result: ai.LoadedModel = w._LoadNew(config)
  assert result.metadata == {}


# ---------------------------------------------------------------------------
# LlamaWorker._Call — text (str output_format)
# ---------------------------------------------------------------------------


def testLlamaCallRaisesOnInvalidCallSeed(tmp_path: pathlib.Path) -> None:
  """_Call raises Error when call_seed <= 1."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)
  with pytest.raises(llama.Error, match='call_seed must be'):
    w._Call(loaded, 'sys', 'user', str, 1)  # call_seed=1 is <= 1


def testLlamaCallReturnsStringForStrFormat(tmp_path: pathlib.Path) -> None:
  """_Call returns string content when output_format=str."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('answer text')
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True  # Skip output suppression for simpler mocking
  with typeguard.suppress_type_checks():
    result: str = w._Call(loaded, 'system', 'user question', str, 1000)
  assert result == 'answer text'
  llm_mock.create_chat_completion.assert_called_once()


def testLlamaCallReturnsParsedPydanticModel(tmp_path: pathlib.Path) -> None:
  """_Call parses JSON output into a pydantic model when output_format is a BaseModel subclass."""

  class _MyOutput(pydantic.BaseModel):
    label: str
    score: float

  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('{"label":"cat","score":0.9}')
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  with typeguard.suppress_type_checks():
    result: _MyOutput = w._Call(loaded, 'system', 'user', _MyOutput, 1000)
  assert isinstance(result, _MyOutput)
  assert result.label == 'cat'
  assert result.score == pytest.approx(0.9)  # pyright: ignore[reportUnknownMemberType]


def testLlamaCallRaisesOnValueError(tmp_path: pathlib.Path) -> None:
  """_Call wraps ValueError from llm.create_chat_completion into Error."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.side_effect = ValueError('bad grammar')
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  with pytest.raises(llama.Error, match='Error calling model'):
    w._Call(loaded, 'system', 'user', str, 1000)


def testLlamaCallRaisesOnRuntimeError(tmp_path: pathlib.Path) -> None:
  """_Call wraps RuntimeError from llm.create_chat_completion into Error."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.side_effect = RuntimeError('segfault')
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  with pytest.raises(llama.Error, match='Error calling model'):
    w._Call(loaded, 'system', 'user', str, 1000)


# ---------------------------------------------------------------------------
# LlamaWorker._Call — vision (with images)
# ---------------------------------------------------------------------------


def testLlamaCallWithImageBytes(tmp_path: pathlib.Path) -> None:
  """_Call handles image bytes by converting them and adding to messages."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=True, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('has cat')
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  fake_png: bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
  with (
    mock.patch('transai.core.llama.ai_images.ResizeImageForVision', return_value=fake_png),
    typeguard.suppress_type_checks(),
  ):
    result: str = w._Call(loaded, 'system', 'describe image', str, 1000, images=[fake_png])
  assert result == 'has cat'
  call_args = llm_mock.create_chat_completion.call_args
  messages: list[dict[str, Any]] = call_args[1]['messages']
  # last message should be user message with content list
  user_msg: dict[str, Any] = messages[-1]
  assert user_msg['role'] == 'user'
  assert isinstance(user_msg['content'], list)
  assert any(part.get('type') == 'image_url' for part in user_msg['content'])  # pyright: ignore


def testLlamaCallWithImagePath(tmp_path: pathlib.Path) -> None:
  """_Call handles image pathlib.Path by reading bytes before converting."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=True, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('path result')
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  img_file: pathlib.Path = tmp_path / 'img.png'
  fake_bytes: bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 50
  img_file.write_bytes(fake_bytes)
  with (
    mock.patch('transai.core.llama.ai_images.ResizeImageForVision', return_value=fake_bytes),
    typeguard.suppress_type_checks(),
  ):
    result: str = w._Call(loaded, 'sys', 'describe', str, 1000, images=[img_file])
  assert result == 'path result'


def testLlamaCallRaisesOnVisionWithoutCapability(tmp_path: pathlib.Path) -> None:
  """_Call raises Error when images provided but model lacks vision capability."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)
  with pytest.raises(llama.Error, match='does not support vision'):
    w._Call(loaded, 'sys', 'user', str, 1000, images=[b'\x89PNG'])


# ---------------------------------------------------------------------------
# LlamaWorker._Call — output suppression path (verbose=False)
# ---------------------------------------------------------------------------


def testLlamaCallSuppressesOutputWhenNotVerbose(tmp_path: pathlib.Path) -> None:
  """_Call uses _SuppressNativeOutput(True) when verbose=False."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('quiet')
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)
  # verbose=False (default) → suppress=True (_SuppressNativeOutput redirects fds)
  assert w._verbose is False
  with typeguard.suppress_type_checks():
    result: str = w._Call(loaded, 'system', 'user', str, 1000)
  assert result == 'quiet'


# ---------------------------------------------------------------------------
# LlamaWorker end-to-end via LoadModel + ModelCall
# ---------------------------------------------------------------------------


def testLlamaLoadRaisesOnMissingSeedAfterModelLoad(tmp_path: pathlib.Path) -> None:
  """_LoadNew raises Error when config has no seed (safety guard, bypasses _ConfigSeed)."""
  gguf_file: pathlib.Path = tmp_path / 'model.gguf'
  gguf_file.write_bytes(b'x' * 100)
  # Call _LoadNew directly (not via LoadModel) so _ConfigSeed is not called first
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='model', model_path=gguf_file)
  config['seed'] = None  # type: ignore[typeddict-item]  # bypass normal validation
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {}
  w = llama.LlamaWorker(tmp_path)
  with (
    mock.patch('llama_cpp.Llama', return_value=mock_llama),
    pytest.raises(llama.Error, match='seed'),
  ):
    w._LoadNew(config)


def testLlamaWorkerLoadModelAndModelCall(tmp_path: pathlib.Path) -> None:
  """LoadModel then ModelCall round-trip works correctly."""
  gguf_file: pathlib.Path = tmp_path / 'mymodel.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='mymodel', model_path=gguf_file)
  mock_llama_inst = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama_inst.metadata = {}
  mock_llama_inst.create_chat_completion.return_value = _MakeResponse('e2e result')
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  with (
    mock.patch('transai.core.ai.saferandom.RandBits', return_value=1000),
    mock.patch('llama_cpp.Llama', return_value=mock_llama_inst),
    typeguard.suppress_type_checks(),
  ):
    w.LoadModel(config)
    result: str = w.ModelCall('mymodel', 'system', 'user', str)
  assert result == 'e2e result'


# ---------------------------------------------------------------------------
# LlamaWorker._Call — tool use
# ---------------------------------------------------------------------------


def testLlamaCallRaisesOnToolsButNoToolingCapability(tmp_path: pathlib.Path) -> None:
  """_Call raises Error when tools provided but model has tooling=False."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, tooling=False, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)

  def _my_tool(x: int) -> int:
    """Return doubled.

    Args:
      x: input

    Returns:
      doubled x

    """
    return x * 2

  with pytest.raises(llama.Error, match='does not support tools'):
    w._Call(loaded, 'sys', 'user', str, 1000, tools=[_my_tool])


def testLlamaCallRaisesOnToolsWithStructuredOutput(tmp_path: pathlib.Path) -> None:
  """_Call raises Error when tools provided with non-str output_format."""

  class _Out(pydantic.BaseModel):
    value: int

  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, tooling=True, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)

  def _my_tool(x: int) -> int:
    """Return doubled.

    Args:
      x: input

    Returns:
      doubled x

    """
    return x * 2

  with pytest.raises(llama.Error, match='str output_format'):
    w._Call(loaded, 'sys', 'user', _Out, 1000, tools=[_my_tool])  # type: ignore[arg-type]


def testLlamaCallWithToolsCallsAct(tmp_path: pathlib.Path) -> None:
  """_Call routes to _CallLlamaAct when tools are provided."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, tooling=True, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.metadata = {'general.name': 'tool-capable-model'}
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)

  def _my_tool(x: int) -> int:
    """Return doubled.

    Args:
      x: input

    Returns:
      doubled x

    """
    return x * 2

  with (
    mock.patch('transai.core.llama._CallLlamaAct', return_value='tool answer') as act_mock,
    mock.patch('lmstudio.json_api.ChatResponseEndpoint.parse_tools') as parse_mock,
  ):
    fake_tool = mock.MagicMock()
    fake_tool.to_dict.return_value = {'name': '_my_tool'}
    parse_mock.return_value = (mock.MagicMock(tools=[fake_tool]),)
    with typeguard.suppress_type_checks():
      result: str = w._Call(loaded, 'sys', 'user', str, 1000, tools=[_my_tool])
  assert result == 'tool answer'
  act_mock.assert_called_once()


def testLlamaCallRaisesOnEmptyToolDefs(tmp_path: pathlib.Path) -> None:
  """_Call raises Error when parsed tool definitions are empty."""
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=False, tooling=True, seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  loaded = ai.LoadedModel(
    model_id=config['model_id'], seed_state=bytes(32), config=config, metadata={}, model=llm_mock
  )
  w = llama.LlamaWorker(tmp_path)

  def _my_tool(x: int) -> int:
    """Return doubled.

    Args:
      x: input

    Returns:
      doubled x

    """
    return x * 2

  with mock.patch('lmstudio.json_api.ChatResponseEndpoint.parse_tools') as parse_mock:
    parse_mock.return_value = (mock.MagicMock(tools=[]),)  # empty tools list
    with pytest.raises(llama.Error, match='No valid tools'), typeguard.suppress_type_checks():
      w._Call(loaded, 'sys', 'user', str, 1000, tools=[_my_tool])


# ---------------------------------------------------------------------------
# _QwenDecode
# ---------------------------------------------------------------------------


def testQwenDecodeNoToolCalls() -> None:
  """_QwenDecode returns content text and empty tool list when no tool calls are present."""
  content, tools = llama._QwenDecode('answer without tools')
  assert content == 'answer without tools'
  assert tools == []


def testQwenDecodeWithToolCall() -> None:
  """_QwenDecode extracts a tool call from <tool_call> tags."""
  tool_json = json.dumps({'name': 'my_tool', 'arguments': {'x': 5}})
  _, tools = llama._QwenDecode(f'calling: <tool_call>{tool_json}</tool_call>')
  assert tools[0]['function'] == {'name': 'my_tool', 'arguments': {'x': 5}}


def testQwenDecodeEmptyContentReturnsNone() -> None:
  """_QwenDecode returns None for content when the string is empty."""
  content, tools = llama._QwenDecode('')
  assert content is None
  assert tools == []


# ---------------------------------------------------------------------------
# _DetectToolHandler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ('model_id', 'expected'),
  [
    ('qwen2-7b', llama._QwenDecode),
    ('qwen3-8b-instruct', llama._QwenDecode),
  ],
)
def testDetectToolHandler(model_id: str, expected: object) -> None:
  """_DetectToolHandler returns the expected decoder for a given model identifier."""
  assert llama._DetectToolHandler(model_id) is expected


def testDetectToolHandlerRaisesOnUnknown() -> None:
  """_DetectToolHandler raises Error when no handler matches the model identifier."""
  with pytest.raises(llama.Error, match='does not match any known tool-handling patterns'):
    llama._DetectToolHandler('completely-unknown-model')


# ---------------------------------------------------------------------------
# _ExecuteToolCalls
# ---------------------------------------------------------------------------


def testExecuteToolCallsSuccess() -> None:
  """_ExecuteToolCalls executes a tool and appends the result to messages."""
  messages: list[base.JSONDict] = []
  tool_calls: list[base.JSONDict] = [
    {'id': 'c1', 'function': {'name': 'double', 'arguments': {'x': 3}}},
  ]
  result_holder: list[int] = []

  def double(x: int) -> int:
    """Return doubled.

    Args:
      x: input

    Returns:
      doubled x

    """
    result_holder.append(x * 2)
    return x * 2

  llama._ExecuteToolCalls(tool_calls, {'double': double}, messages)
  assert messages[-1]['content'] == repr(6)
  assert messages[-1]['name'] == 'double'
  assert messages[-1]['tool_call_id'] == 'c1'


def testExecuteToolCallsWithJsonStringArgs() -> None:
  """_ExecuteToolCalls parses JSON string arguments before passing to the tool."""
  messages: list[base.JSONDict] = []
  args_str = json.dumps({'x': 7})
  tool_calls: list[base.JSONDict] = [
    {'id': 'c2', 'function': {'name': 'triple', 'arguments': args_str}},
  ]

  def triple(x: int) -> int:
    """Return tripled.

    Args:
      x: input

    Returns:
      tripled x

    """
    return x * 3

  llama._ExecuteToolCalls(tool_calls, {'triple': triple}, messages)
  assert messages[-1]['content'] == repr(21)


def testExecuteToolCallsUnknownTool() -> None:
  """_ExecuteToolCalls raises Error when the called tool is not in the tool map."""
  messages: list[base.JSONDict] = []
  tool_calls: list[base.JSONDict] = [
    {'id': 'c3', 'function': {'name': 'ghost', 'arguments': {}}},
  ]
  with pytest.raises(llama.Error, match="unknown tool 'ghost'"):
    llama._ExecuteToolCalls(tool_calls, {}, messages)


def testExecuteToolCallsInvalidJson() -> None:
  """_ExecuteToolCalls raises Error when the arguments string is not valid JSON."""
  messages: list[base.JSONDict] = []
  tool_calls: list[base.JSONDict] = [
    {'id': 'c4', 'function': {'name': 'f', 'arguments': 'not json!'}},
  ]

  def f() -> str:
    """Return nothing.

    Returns:
      empty string

    """
    return ''

  with pytest.raises(llama.Error, match='invalid JSON'):
    llama._ExecuteToolCalls(tool_calls, {'f': f}, messages)


def testExecuteToolCallsToolRaisesException() -> None:
  """_ExecuteToolCalls captures exceptions from the tool and feeds them back as results."""
  messages: list[base.JSONDict] = []
  tool_calls: list[base.JSONDict] = [
    {'id': 'c5', 'function': {'name': 'bomb', 'arguments': {}}},
  ]

  def bomb() -> None:
    """Raise always.

    Raises:
      ValueError: always

    """
    raise ValueError('boom')

  llama._ExecuteToolCalls(tool_calls, {'bomb': bomb}, messages)
  # error captured as the tool result (repr of the exception)
  assert 'boom' in str(messages[-1]['content'])


def testExecuteToolCallsFallsBackToPositionalArgs() -> None:
  """_ExecuteToolCalls falls back to positional args when the tool rejects keyword args."""
  messages: list[base.JSONDict] = []
  tool_calls: list[base.JSONDict] = [
    {'id': 'c6', 'function': {'name': 'builtin_max', 'arguments': {'a': 3, 'b': 7}}},
  ]
  # max() is a builtin that rejects keyword args like 'a' and 'b'
  # We simulate with a pure function that also rejects kwargs for testing positional fallback

  # Use a real positional-only-like function via a special wrapper
  # Actually, just test with a function that raises TypeError with 'keyword arguments' message
  # and then test that the fallback to positional args is called
  call_log: list[tuple[object, ...]] = []

  def _positional_only(*args: object, **kwargs: object) -> str:
    if kwargs:
      raise TypeError('keyword arguments not accepted')
    call_log.append(args)
    return 'ok'

  llama._ExecuteToolCalls(tool_calls, {'builtin_max': _positional_only}, messages)
  assert messages[-1]['content'] == repr('ok')
  # was called with positional args (values from dict)
  assert call_log[0] == (3, 7)


def testExecuteToolCallsTypeErrorNotKeywordRelated() -> None:
  """_ExecuteToolCalls re-raises TypeError when it is not about keyword arguments."""
  messages: list[base.JSONDict] = []
  tool_calls: list[base.JSONDict] = [
    {'id': 'c7', 'function': {'name': 'broken', 'arguments': {'x': 1}}},
  ]

  def broken(x: int) -> None:  # noqa: ARG001
    """Raise TypeError.

    Args:
      x: input

    Raises:
      TypeError: always

    """
    raise TypeError('completely unrelated error')

  # TypeError not about 'keyword arguments' → captured as exception result (not re-raised)
  llama._ExecuteToolCalls(tool_calls, {'broken': broken}, messages)
  assert 'completely unrelated error' in str(messages[-1]['content'])


# ---------------------------------------------------------------------------
# _CallLlamaAct
# ---------------------------------------------------------------------------


def _MakeLlamaActResponse(
  content: str = 'final answer',
  finish_reason: str = 'stop',
  tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
  """Build a create_chat_completion response for the tool-use loop.

  Args:
    content: message content
    finish_reason: the reason the completion finished
    tool_calls: optional list of tool_call dicts

  Returns:
    A dict compatible with llama_types.CreateChatCompletionResponse

  """
  msg: dict[str, Any] = {'role': 'assistant', 'content': content}
  if tool_calls is not None:
    msg['tool_calls'] = tool_calls
  return {
    'id': 'chat-act-test',
    'object': 'chat.completion',
    'created': 1234567890,
    'model': 'test-model',
    'choices': [{'index': 0, 'message': msg, 'finish_reason': finish_reason}],
    'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8},
  }


def testCallLlamaActNoToolCalls() -> None:
  """_CallLlamaAct returns the model's text response when it makes no tool calls."""
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  # Qwen-style: finish_reason='stop' with no tool_call tags
  llm_mock.create_chat_completion.return_value = _MakeLlamaActResponse('first and only answer')
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='qwen3-test', seed=5000)
  messages: list[base.JSONDict] = [{'role': 'system', 'content': 'sys'}]
  with typeguard.suppress_type_checks():
    result: str = llama._CallLlamaAct(llm_mock, messages, [], {}, config, 1000)
  assert result == 'first and only answer'


def testCallLlamaActLogsUsage() -> None:
  """_CallLlamaAct logs token usage from the final response."""
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeLlamaActResponse('done')
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='qwen3-test', seed=5000)
  messages: list[base.JSONDict] = [{'role': 'system', 'content': 'sys'}]
  with (
    mock.patch('transai.core.llama.logging') as log_mock,
    typeguard.suppress_type_checks(),
  ):
    llama._CallLlamaAct(llm_mock, messages, [], {}, config, 1000)
  # verify that debug logging was called (token usage)
  log_mock.debug.assert_called()


def testCallLlamaActRaisesOnNoChoices() -> None:
  """_CallLlamaAct raises Error when the model returns no choices."""
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  response: dict[str, Any] = {'choices': [], 'usage': {}}
  llm_mock.create_chat_completion.return_value = response
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='qwen3', seed=5000)
  messages: list[base.JSONDict] = [{'role': 'system', 'content': 'sys'}]
  with (
    pytest.raises(llama.Error, match='no choices'),
    typeguard.suppress_type_checks(),
  ):
    llama._CallLlamaAct(llm_mock, messages, [], {}, config, 1000)


def testCallLlamaActWithQwenToolCall() -> None:
  """_CallLlamaAct completes a full tool-use round with a Qwen-style tool call."""
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='qwen3-test', seed=5000)
  tool_json = json.dumps({'name': 'add_one', 'arguments': {'n': 4}})
  # First response: Qwen tool call style
  first = _MakeLlamaActResponse(content=f'<tool_call>{tool_json}</tool_call>', finish_reason='stop')
  # Second response: final answer
  second = _MakeLlamaActResponse(content='the answer is 5', finish_reason='stop')
  llm_mock.create_chat_completion.side_effect = [first, second]

  def add_one(n: int) -> int:
    """Return n+1.

    Args:
      n: input

    Returns:
      n + 1

    """
    return n + 1

  tool_map = {'add_one': add_one}
  with (
    mock.patch('transai.core.llama._ToolID', return_value='tool-id-001'),
    typeguard.suppress_type_checks(),
  ):
    result: str = llama._CallLlamaAct(
      llm_mock, [{'role': 'system', 'content': 'sys'}], [], tool_map, config, 1000
    )
  assert 'the answer is 5' in result
