# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""llama.py unittest.

Run with:
  poetry run pytest -vvv tests/core/llama_test.py
"""

from __future__ import annotations

import base64
import pathlib
from typing import Any
from unittest import mock

import llama_cpp
import pydantic
import pytest
import typeguard
from llama_cpp import llama_chat_format

from tests.core import ai_test
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


def testExtractContentSuccessLengthFinishReason() -> None:
  """_ExtractContent accepts 'length' as a valid finish_reason."""
  with typeguard.suppress_type_checks():
    assert llama._ExtractContent(_MakeResponse('partial', 'length')) == 'partial'  # type: ignore[arg-type]


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
  config: ai.AIModelConfig = ai_test.MakeConfig(model_path=gguf_file)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {'general.name': 'test'}
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    loaded_config, _, model = w._Load(config)
  assert loaded_config['model_path'] == gguf_file
  assert loaded_config['clip_path'] is None
  assert loaded_config['vision'] is False
  assert model is mock_llama


def testLlamaWorkerLoadSearchesForModelWhenNoPath(tmp_path: pathlib.Path) -> None:
  """_Load searches for a model directory when model_path is None."""
  model_dir: pathlib.Path = tmp_path / 'mymodel'
  model_dir.mkdir()
  gguf_file: pathlib.Path = model_dir / 'mymodel.gguf'
  gguf_file.write_bytes(b'model data' * 10)
  config: ai.AIModelConfig = ai_test.MakeConfig(model_id='mymodel')
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {'general.name': 'test'}
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    loaded_config, _, model = w._Load(config)
  assert loaded_config['model_path'] == gguf_file
  assert model is mock_llama


def testLlamaWorkerLoadSetsVisionTrueWithClip(tmp_path: pathlib.Path) -> None:
  """_Load sets vision=True when a clip file is provided."""
  gguf_file: pathlib.Path = tmp_path / 'qwen3-vl-model.gguf'
  clip_file: pathlib.Path = tmp_path / 'mmproj-clip.gguf'
  gguf_file.write_bytes(b'model' * 20)
  clip_file.write_bytes(b'clip' * 10)
  config: ai.AIModelConfig = ai_test.MakeConfig(
    model_id='qwen3-vl', model_path=gguf_file, clip_path=clip_file, vision=True
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
    loaded_config, _, _ = w._Load(config)
  assert loaded_config['vision'] is True
  mock_handler_cls.assert_called_once_with(clip_model_path=str(clip_file.resolve()))


def testLlamaWorkerLoadRaisesIfVisionRequestedButNoClip(tmp_path: pathlib.Path) -> None:
  """_Load raises Error when vision=True but no clip file found."""
  gguf_file: pathlib.Path = tmp_path / 'no-vis.gguf'
  gguf_file.write_bytes(b'model' * 20)
  config: ai.AIModelConfig = ai_test.MakeConfig(
    model_id='no-vis', model_path=gguf_file, vision=True
  )
  w = llama.LlamaWorker(tmp_path)
  with pytest.raises(llama.Error, match='Vision requested'):
    w._Load(config)


def testLlamaWorkerLoadRaisesOnUnknownVisionHandler(tmp_path: pathlib.Path) -> None:
  """_Load raises Error when clip found but no handler detected."""
  gguf_file: pathlib.Path = tmp_path / 'unknown-vis.gguf'
  clip_file: pathlib.Path = tmp_path / 'mmproj.gguf'
  gguf_file.write_bytes(b'model' * 20)
  clip_file.write_bytes(b'clip' * 10)
  config: ai.AIModelConfig = ai_test.MakeConfig(model_path=gguf_file, clip_path=clip_file)
  w = llama.LlamaWorker(tmp_path)
  with (
    mock.patch('transai.core.llama._DetectVisionHandler', return_value=None),
    pytest.raises(llama.Error, match='no vision handler'),
  ):
    w._Load(config)


def testLlamaWorkerLoadUsesSpecTokens(tmp_path: pathlib.Path) -> None:
  """_Load creates a draft model when spec_tokens > 0."""
  gguf_file: pathlib.Path = tmp_path / 'mymodel.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai_test.MakeConfig(model_path=gguf_file, spec_tokens=5)
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
    w._Load(config)
  draft_cls.assert_called_once_with(num_pred_tokens=5)


def testLlamaWorkerLoadSetsToolingFromMetadata(tmp_path: pathlib.Path) -> None:
  """_Load sets tooling=True when metadata contains tooling keywords."""
  gguf_file: pathlib.Path = tmp_path / 'tool-model.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai_test.MakeConfig(model_path=gguf_file)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {'model.type': 'functionary-chat'}
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    loaded_config, _, _ = w._Load(config)
  assert loaded_config['tooling'] is True


def testLlamaWorkerLoadSetsReasoningFromMetadata(tmp_path: pathlib.Path) -> None:
  """_Load sets reasoning=True when metadata contains reasoning keywords."""
  gguf_file: pathlib.Path = tmp_path / 'reason-model.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai_test.MakeConfig(model_path=gguf_file)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = {'model.type': 'deepseek-r1-reasoning'}
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    loaded_config, _, _ = w._Load(config)
  assert loaded_config['reasoning'] is True


def testLlamaWorkerLoadHandlesNoneMetadata(tmp_path: pathlib.Path) -> None:
  """_Load handles the case where llm.metadata is None."""
  gguf_file: pathlib.Path = tmp_path / 'mymodel.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai_test.MakeConfig(model_path=gguf_file)
  w = llama.LlamaWorker(tmp_path)
  mock_llama = mock.MagicMock(spec=llama_cpp.Llama)
  mock_llama.metadata = None
  with mock.patch('llama_cpp.Llama', return_value=mock_llama):
    _, metadata, _ = w._Load(config)
  assert metadata == {}


# ---------------------------------------------------------------------------
# LlamaWorker._Call — text (str output_format)
# ---------------------------------------------------------------------------


def testLlamaCallReturnsStringForStrFormat(tmp_path: pathlib.Path) -> None:
  """_Call returns string content when output_format=str."""
  config: ai.AIModelConfig = ai_test.MakeConfig(vision=False)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('answer text')
  loaded: ai.LoadedModel = (config, {}, llm_mock)
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True  # Skip output suppression for simpler mocking
  with typeguard.suppress_type_checks():
    result: str = w._Call(loaded, 'system', 'user question', str)
  assert result == 'answer text'
  llm_mock.create_chat_completion.assert_called_once()


def testLlamaCallReturnsParsedPydanticModel(tmp_path: pathlib.Path) -> None:
  """_Call parses JSON output into a pydantic model when output_format is a BaseModel subclass."""

  class _MyOutput(pydantic.BaseModel):
    label: str
    score: float

  config: ai.AIModelConfig = ai_test.MakeConfig(vision=False)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('{"label":"cat","score":0.9}')
  loaded: ai.LoadedModel = (config, {}, llm_mock)
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  with typeguard.suppress_type_checks():
    result: _MyOutput = w._Call(loaded, 'system', 'user', _MyOutput)
  assert isinstance(result, _MyOutput)
  assert result.label == 'cat'
  assert result.score == pytest.approx(0.9)  # pyright: ignore[reportUnknownMemberType]


def testLlamaCallRaisesOnValueError(tmp_path: pathlib.Path) -> None:
  """_Call wraps ValueError from llm.create_chat_completion into Error."""
  config: ai.AIModelConfig = ai_test.MakeConfig(vision=False)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.side_effect = ValueError('bad grammar')
  loaded: ai.LoadedModel = (config, {}, llm_mock)
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  with pytest.raises(llama.Error, match='Error calling model'):
    w._Call(loaded, 'system', 'user', str)


def testLlamaCallRaisesOnRuntimeError(tmp_path: pathlib.Path) -> None:
  """_Call wraps RuntimeError from llm.create_chat_completion into Error."""
  config: ai.AIModelConfig = ai_test.MakeConfig(vision=False)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.side_effect = RuntimeError('segfault')
  loaded: ai.LoadedModel = (config, {}, llm_mock)
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  with pytest.raises(llama.Error, match='Error calling model'):
    w._Call(loaded, 'system', 'user', str)


# ---------------------------------------------------------------------------
# LlamaWorker._Call — vision (with images)
# ---------------------------------------------------------------------------


def testLlamaCallWithImageBytes(tmp_path: pathlib.Path) -> None:
  """_Call handles image bytes by converting them and adding to messages."""
  config: ai.AIModelConfig = ai_test.MakeConfig(vision=True)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('has cat')
  loaded: ai.LoadedModel = (config, {}, llm_mock)
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  fake_png: bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
  with (
    mock.patch('transai.core.llama.ai_images.ResizeImageForVision', return_value=fake_png),
    typeguard.suppress_type_checks(),
  ):
    result: str = w._Call(loaded, 'system', 'describe image', str, images=[fake_png])
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
  config: ai.AIModelConfig = ai_test.MakeConfig(vision=True)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('path result')
  loaded: ai.LoadedModel = (config, {}, llm_mock)
  w = llama.LlamaWorker(tmp_path)
  w._verbose = True
  img_file: pathlib.Path = tmp_path / 'img.png'
  fake_bytes: bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 50
  img_file.write_bytes(fake_bytes)
  with (
    mock.patch('transai.core.llama.ai_images.ResizeImageForVision', return_value=fake_bytes),
    typeguard.suppress_type_checks(),
  ):
    result: str = w._Call(loaded, 'sys', 'describe', str, images=[img_file])
  assert result == 'path result'


def testLlamaCallRaisesOnVisionWithoutCapability(tmp_path: pathlib.Path) -> None:
  """_Call raises Error when images provided but model lacks vision capability."""
  config: ai.AIModelConfig = ai_test.MakeConfig(vision=False)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  loaded: ai.LoadedModel = (config, {}, llm_mock)
  w = llama.LlamaWorker(tmp_path)
  with pytest.raises(llama.Error, match='does not support vision'):
    w._Call(loaded, 'sys', 'user', str, images=[b'\x89PNG'])


# ---------------------------------------------------------------------------
# LlamaWorker._Call — output suppression path (verbose=False)
# ---------------------------------------------------------------------------


def testLlamaCallSuppressesOutputWhenNotVerbose(tmp_path: pathlib.Path) -> None:
  """_Call uses _SuppressNativeOutput(True) when verbose=False."""
  config: ai.AIModelConfig = ai_test.MakeConfig(vision=False)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  llm_mock.create_chat_completion.return_value = _MakeResponse('quiet')
  loaded: ai.LoadedModel = (config, {}, llm_mock)
  w = llama.LlamaWorker(tmp_path)
  # verbose=False (default) → suppress=True (_SuppressNativeOutput redirects fds)
  assert w._verbose is False
  with typeguard.suppress_type_checks():
    result: str = w._Call(loaded, 'system', 'user', str)
  assert result == 'quiet'


# ---------------------------------------------------------------------------
# LlamaWorker end-to-end via LoadModel + ModelCall
# ---------------------------------------------------------------------------


def testLlamaWorkerLoadModelAndModelCall(tmp_path: pathlib.Path) -> None:
  """LoadModel then ModelCall round-trip works correctly."""
  gguf_file: pathlib.Path = tmp_path / 'mymodel.gguf'
  gguf_file.write_bytes(b'x' * 100)
  config: ai.AIModelConfig = ai_test.MakeConfig(model_id='mymodel', model_path=gguf_file)
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
