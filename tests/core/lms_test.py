# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""lms.py unittest.

Run with:
  poetry run pytest -vvv tests/core/lms_test.py
"""

from __future__ import annotations

import pathlib
from unittest import mock

import lmstudio
import pydantic
import pytest

from transai.core import ai, lms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _MakeLMSWorker(
  *,
  api_host: str = '127.0.0.1:1234',
  loaded_models: list[str] | None = None,
) -> tuple[lms.LMStudioWorker, mock.MagicMock]:
  """Create an LMStudioWorker with mocked lmstudio.Client.

  Args:
    api_host: the API host to return from find_default_local_api_host
    loaded_models: list of model identifiers to return from llm.list_loaded()

  Returns:
    (worker_instance, client_mock)

  """
  client_mock = mock.MagicMock()
  loaded: list[mock.MagicMock] = []
  for mid in loaded_models or []:
    m = mock.MagicMock()
    m.identifier = mid
    loaded.append(m)
  client_mock.llm.list_loaded.return_value = loaded
  # Use a single class mock: find_default_local_api_host() returns the host string,
  # and instantiation (Client(host)) returns client_mock
  client_cls_mock = mock.MagicMock()
  client_cls_mock.find_default_local_api_host.return_value = api_host
  client_cls_mock.return_value = client_mock
  with (
    mock.patch.object(lmstudio, 'Client', client_cls_mock),
    mock.patch.object(lmstudio, 'set_sync_api_timeout'),
  ):
    worker = lms.LMStudioWorker()
  return worker, client_mock


# ---------------------------------------------------------------------------
# LMStudioWorker.__init__
# ---------------------------------------------------------------------------


def testLMSWorkerInitRaisesWhenNoApiHost() -> None:
  """LMStudioWorker raises Error when no LM Studio API host is found."""
  client_cls_mock = mock.MagicMock()
  client_cls_mock.find_default_local_api_host.return_value = None
  with (
    mock.patch.object(lmstudio, 'Client', client_cls_mock),
    mock.patch.object(lmstudio, 'set_sync_api_timeout'),
    pytest.raises(lms.Error, match='No LM Studio API server'),
  ):
    lms.LMStudioWorker()


def testLMSWorkerInitRaisesWhenHostNotLoopback() -> None:
  """LMStudioWorker raises Error when host is not a loopback address."""
  client_cls_mock = mock.MagicMock()
  client_cls_mock.find_default_local_api_host.return_value = '192.168.1.5:1234'
  with (
    mock.patch.object(lmstudio, 'Client', client_cls_mock),
    mock.patch.object(lmstudio, 'set_sync_api_timeout'),
    pytest.raises(lms.Error, match='not a loopback address'),
  ):
    lms.LMStudioWorker()


def testLMSWorkerInitAcceptsLocalhostHost() -> None:
  """LMStudioWorker accepts 'localhost:...' as a valid host."""
  worker, _client = _MakeLMSWorker(api_host='localhost:1234')
  assert worker._api_host == 'localhost:1234'


def testLMSWorkerInitUnloadsExistingModels() -> None:
  """LMStudioWorker unloads any models already loaded in LM Studio."""
  _worker, client_mock = _MakeLMSWorker(loaded_models=['model-a', 'model-b'])
  assert client_mock.llm.unload.call_count == 2
  client_mock.llm.unload.assert_any_call('model-a')
  client_mock.llm.unload.assert_any_call('model-b')


def testLMSWorkerInitNoModelsToUnload() -> None:
  """LMStudioWorker init works fine when no models are currently loaded."""
  _worker, client_mock = _MakeLMSWorker(loaded_models=[])
  client_mock.llm.unload.assert_not_called()


def testLMSWorkerInitSetsTimeout() -> None:
  """LMStudioWorker sets sync API timeout to None (infinite)."""
  client_cls_mock = mock.MagicMock()
  client_cls_mock.find_default_local_api_host.return_value = '127.0.0.1:1'
  client_cls_mock.return_value = mock.MagicMock()
  with (
    mock.patch.object(lmstudio, 'Client', client_cls_mock),
    mock.patch.object(lmstudio, 'set_sync_api_timeout') as set_timeout_mock,
  ):
    lms.LMStudioWorker()
  set_timeout_mock.assert_called_once_with(None)


# ---------------------------------------------------------------------------
# LMStudioWorker.Close
# ---------------------------------------------------------------------------


def testLMSWorkerCloseCallsClientClose() -> None:
  """Close() must call self._client.close()."""
  worker, client_mock = _MakeLMSWorker()
  worker.Close()
  client_mock.close.assert_called_once()


def testLMSWorkerCloseAlsoClearsLoadedModels() -> None:
  """Close() must clear _loaded_models via super().Close()."""
  worker, _client = _MakeLMSWorker()
  lm_mock = mock.MagicMock(spec=lmstudio.LLM)
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  worker._loaded_models['test-model'] = (config, {}, lm_mock)
  worker.Close()
  assert worker._loaded_models == {}


# ---------------------------------------------------------------------------
# LMStudioWorker._Load
# ---------------------------------------------------------------------------


def _MakeModelInfo(
  *,
  model_key: str = 'test-model',
  vision: bool = False,
  tooling: bool = False,
  max_context: int = 2048,
) -> mock.MagicMock:
  """Build a mock lmstudio.LlmInstanceInfo.

  Returns:
    MagicMock with spec=lmstudio.LlmInstanceInfo and fields set according to arguments

  """
  info = mock.MagicMock(spec=lmstudio.LlmInstanceInfo)
  info.model_key = model_key
  info.vision = vision
  info.trained_for_tool_use = tooling
  info.max_context_length = max_context
  info.to_dict.return_value = {'model_key': model_key, 'vision': vision, 'tooling': tooling}
  return info


def testLMSWorkerLoadSuccessBasic() -> None:
  """_Load succeeds for a basic text model."""
  worker, client_mock = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(context=1024)
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  model_info: mock.MagicMock = _MakeModelInfo()
  lm_model_mock.get_info.return_value = model_info
  lm_model_mock.get_context_length.return_value = 2048
  client_mock.llm.load_new_instance.return_value = lm_model_mock
  with mock.patch('lmstudio.LlmLoadModelConfigDict', return_value={}):
    loaded_config, metadata, model = worker._Load(config)
  assert loaded_config['model_id'] == 'test-model'
  assert model is lm_model_mock
  assert metadata == {'model_key': 'test-model', 'vision': False, 'tooling': False}


def testLMSWorkerLoadUpdatesModelIdFromInfo() -> None:
  """_Load replaces model_id with the key returned by model_info.model_key."""
  worker, client_mock = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='short-name', context=1024)
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  model_info: mock.MagicMock = _MakeModelInfo(model_key='canonical/model-key')
  lm_model_mock.get_info.return_value = model_info
  lm_model_mock.get_context_length.return_value = 4096
  client_mock.llm.load_new_instance.return_value = lm_model_mock
  with mock.patch('lmstudio.LlmLoadModelConfigDict', return_value={}):
    loaded_config, _meta, _model = worker._Load(config)
  assert loaded_config['model_id'] == 'canonical/model-key'


def testLMSWorkerLoadSetsVisionFromModelInfo() -> None:
  """_Load propagates vision and tooling from model_info into config."""
  worker, client_mock = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=True, tooling=False, context=1024)
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  model_info: mock.MagicMock = _MakeModelInfo(vision=True, tooling=True)
  lm_model_mock.get_info.return_value = model_info
  lm_model_mock.get_context_length.return_value = 4096
  client_mock.llm.load_new_instance.return_value = lm_model_mock
  with mock.patch('lmstudio.LlmLoadModelConfigDict', return_value={}):
    loaded_config, _meta, _model = worker._Load(config)
  assert loaded_config['vision'] is True
  assert loaded_config['tooling'] is True


def testLMSWorkerLoadRaisesIfModelInfoNotLlmInstanceInfo() -> None:
  """_Load raises Error if get_info() doesn't return an LlmInstanceInfo."""
  worker, client_mock = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  lm_model_mock.get_info.return_value = 'not-an-instance-info'
  client_mock.llm.load_new_instance.return_value = lm_model_mock
  with (
    mock.patch('lmstudio.LlmLoadModelConfigDict', return_value={}),
    pytest.raises(lms.Error, match='not a valid LLM instance'),
  ):
    worker._Load(config)


def testLMSWorkerLoadRaisesIfVisionRequestedButNotSupported() -> None:
  """_Load raises Error when vision=True but model doesn't support it."""
  worker, client_mock = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=True)
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  model_info: mock.MagicMock = _MakeModelInfo(vision=False)
  lm_model_mock.get_info.return_value = model_info
  lm_model_mock.get_context_length.return_value = 4096
  client_mock.llm.load_new_instance.return_value = lm_model_mock
  with (
    mock.patch('lmstudio.LlmLoadModelConfigDict', return_value={}),
    pytest.raises(lms.Error, match='not a vision model'),
  ):
    worker._Load(config)


def testLMSWorkerLoadRaisesIfToolingRequestedButNotSupported() -> None:
  """_Load raises Error when tooling=True but model doesn't support it."""
  worker, client_mock = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(tooling=True)
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  model_info: mock.MagicMock = _MakeModelInfo(tooling=False)
  lm_model_mock.get_info.return_value = model_info
  lm_model_mock.get_context_length.return_value = 4096
  client_mock.llm.load_new_instance.return_value = lm_model_mock
  with (
    mock.patch('lmstudio.LlmLoadModelConfigDict', return_value={}),
    pytest.raises(lms.Error, match='not trained for tool use'),
  ):
    worker._Load(config)


def testLMSWorkerLoadRaisesIfContextLengthInsufficient() -> None:
  """_Load raises Error when model's context length < requested context."""
  worker, client_mock = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(context=8192)
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  model_info: mock.MagicMock = _MakeModelInfo(max_context=16384)
  lm_model_mock.get_info.return_value = model_info
  lm_model_mock.get_context_length.return_value = 512  # too small
  client_mock.llm.load_new_instance.return_value = lm_model_mock
  with (
    mock.patch('lmstudio.LlmLoadModelConfigDict', return_value={}),
    pytest.raises(lms.Error, match='insufficient context length'),
  ):
    worker._Load(config)


def testLMSWorkerLoadRaisesOnReasoning() -> None:
  """_Load raises Error when config.reasoning=True (not supported)."""
  worker, _client = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(reasoning=True)
  with pytest.raises(lms.Error, match='reasoning is not supported'):
    worker._Load(config)


def testLMSWorkerLoadWarnsOnIgnoredFields() -> None:
  """_Load emits warnings for model_path, kv_cache, flash, gpu_layers, spec_tokens."""
  worker, client_mock = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(
    model_path=pathlib.Path('/some/path.gguf'),
    kv_cache=8,
    flash=True,
    gpu_layers=10,
    spec_tokens=3,
    context=1024,
  )
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  model_info: mock.MagicMock = _MakeModelInfo()
  lm_model_mock.get_info.return_value = model_info
  lm_model_mock.get_context_length.return_value = 4096
  client_mock.llm.load_new_instance.return_value = lm_model_mock
  with (
    mock.patch.object(lmstudio, 'LlmLoadModelConfigDict', return_value={}),
    mock.patch('transai.core.lms.logging') as log_mock,
  ):
    worker._Load(config)
  # model_path + kv_cache + (flash | gpu_layers | spec_tokens) = at least 3 warnings
  assert log_mock.warning.call_count >= 3


# ---------------------------------------------------------------------------
# LMStudioWorker._Call — text
# ---------------------------------------------------------------------------


def _MakePredictionResult(
  content: str = 'result',
  stop_reason: str = 'eosFound',
  parsed: object = None,
) -> mock.MagicMock:
  """Build a mock lmstudio.PredictionResult.

  Returns:
    MagicMock with spec=lmstudio.PredictionResult and fields set according to arguments

  """
  result = mock.MagicMock()  # no spec: avoids attribute-access issues
  result.content = content
  result.stats.stop_reason = stop_reason
  result.stats.predicted_tokens_count = 10
  result.stats.time_to_first_token_sec = 0.5
  result.parsed = parsed
  return result


def testLMSCallReturnsStringContent() -> None:
  """_Call returns result.content when output_format=str."""
  worker, _client = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  pred_result: mock.MagicMock = _MakePredictionResult('hello world')
  lm_model_mock.respond.return_value = pred_result
  loaded: ai.LoadedModel = (config, {}, lm_model_mock)
  with (
    mock.patch('lmstudio.LlmPredictionConfigDict', return_value={}),
    mock.patch('lmstudio.Chat') as chat_cls,
  ):
    chat_mock = mock.MagicMock()
    chat_cls.return_value = chat_mock
    result: str = worker._Call(loaded, 'system', 'user question', str)
  assert result == 'hello world'
  chat_mock.add_user_message.assert_called_once_with('user question', images=None)


def testLMSCallReturnsParsedPydanticModel() -> None:
  """_Call parses result.parsed when output_format is a BaseModel subclass."""

  class _MyOutput(pydantic.BaseModel):
    category: str
    confidence: float

  worker, _client = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  parsed_data: dict[str, str | float] = {'category': 'dog', 'confidence': 0.95}
  pred_result: mock.MagicMock = _MakePredictionResult('ignored', parsed=parsed_data)
  lm_model_mock.respond.return_value = pred_result
  loaded: ai.LoadedModel = (config, {}, lm_model_mock)
  with (
    mock.patch('lmstudio.LlmPredictionConfigDict', return_value={}),
    mock.patch('lmstudio.Chat') as chat_cls,
  ):
    chat_cls.return_value = mock.MagicMock()
    result = worker._Call(loaded, 'system', 'user', _MyOutput)
  assert isinstance(result, _MyOutput)
  assert result.category == 'dog'
  assert result.confidence == pytest.approx(0.95)  # pyright: ignore[reportUnknownMemberType]


def testLMSCallRaisesOnLMStudioServerError() -> None:
  """_Call wraps lmstudio.LMStudioServerError into Error."""
  worker, _client = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  lm_model_mock.respond.side_effect = lmstudio.LMStudioServerError('server error')
  loaded: ai.LoadedModel = (config, {}, lm_model_mock)
  with (
    mock.patch('lmstudio.LlmPredictionConfigDict', return_value={}),
    mock.patch('lmstudio.Chat') as chat_cls,
  ):
    chat_cls.return_value = mock.MagicMock()
    with pytest.raises(lms.Error, match='Error calling model'):
      worker._Call(loaded, 'system', 'user', str)


def testLMSCallRaisesOnUnexpectedStopReason() -> None:
  """_Call raises Error when stop_reason is not 'eosFound'."""
  worker, _client = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  pred_result: mock.MagicMock = _MakePredictionResult('partial', stop_reason='maxTokens')
  lm_model_mock.respond.return_value = pred_result
  loaded: ai.LoadedModel = (config, {}, lm_model_mock)
  with (
    mock.patch('lmstudio.LlmPredictionConfigDict', return_value={}),
    mock.patch('lmstudio.Chat') as chat_cls,
  ):
    chat_cls.return_value = mock.MagicMock()
    with pytest.raises(lms.Error, match='Unexpected stop reason'):
      worker._Call(loaded, 'system', 'user', str)


def testLMSCallWithImageBytes() -> None:
  """_Call passes prepared images to chat.add_user_message."""
  worker, _client = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=True)
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  pred_result: mock.MagicMock = _MakePredictionResult('vision result')
  lm_model_mock.respond.return_value = pred_result
  loaded: ai.LoadedModel = (config, {}, lm_model_mock)
  fake_bytes = b'\x89PNG'
  fake_prepared = mock.MagicMock()
  with (
    mock.patch('lmstudio.LlmPredictionConfigDict', return_value={}),
    mock.patch('lmstudio.Chat') as chat_cls,
    mock.patch('lmstudio.prepare_image', return_value=fake_prepared) as prepare_mock,
  ):
    chat_mock = mock.MagicMock()
    chat_cls.return_value = chat_mock
    result: str = worker._Call(loaded, 'system', 'describe', str, images=[fake_bytes])
  assert result == 'vision result'
  prepare_mock.assert_called_once_with(fake_bytes)
  _call_args, call_kwargs = chat_mock.add_user_message.call_args
  assert call_kwargs['images'] == [fake_prepared]


def testLMSCallWithImagePaths() -> None:
  """_Call handles pathlib.Path images correctly (passed to prepare_image)."""
  worker, _client = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(vision=True)
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  pred_result: mock.MagicMock = _MakePredictionResult('path vision result')
  lm_model_mock.respond.return_value = pred_result
  loaded: ai.LoadedModel = (config, {}, lm_model_mock)
  img_path = pathlib.Path('/fake/image.png')
  fake_prepared = mock.MagicMock()
  with (
    mock.patch('lmstudio.LlmPredictionConfigDict', return_value={}),
    mock.patch('lmstudio.Chat') as chat_cls,
    mock.patch('lmstudio.prepare_image', return_value=fake_prepared) as prepare_mock,
  ):
    chat_mock = mock.MagicMock()
    chat_cls.return_value = chat_mock
    result: str = worker._Call(loaded, 'system', 'describe', str, images=[img_path])
  assert result == 'path vision result'
  prepare_mock.assert_called_once_with(img_path)


def testLMSCallNoImages() -> None:
  """_Call passes images=None when no images provided."""
  worker, _client = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  pred_result: mock.MagicMock = _MakePredictionResult('text only')
  lm_model_mock.respond.return_value = pred_result
  loaded: ai.LoadedModel = (config, {}, lm_model_mock)
  with (
    mock.patch('lmstudio.LlmPredictionConfigDict', return_value={}),
    mock.patch('lmstudio.Chat') as chat_cls,
  ):
    chat_mock = mock.MagicMock()
    chat_cls.return_value = chat_mock
    result: str = worker._Call(loaded, 'system', 'user', str)
  assert result == 'text only'
  _args, kwargs = chat_mock.add_user_message.call_args
  assert kwargs['images'] is None


# ---------------------------------------------------------------------------
# End-to-end via LoadModel + ModelCall
# ---------------------------------------------------------------------------


def testLMSWorkerLoadModelAndModelCall() -> None:
  """LoadModel then ModelCall round-trip works correctly."""
  worker, client_mock = _MakeLMSWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='my-lms-model', context=1024)
  lm_model_mock = mock.MagicMock(spec=lmstudio.LLM)
  model_info: mock.MagicMock = _MakeModelInfo(model_key='my-lms-model')
  lm_model_mock.get_info.return_value = model_info
  lm_model_mock.get_context_length.return_value = 4096
  pred_result: mock.MagicMock = _MakePredictionResult('e2e lms result')
  lm_model_mock.respond.return_value = pred_result
  client_mock.llm.load_new_instance.return_value = lm_model_mock
  with (
    mock.patch('lmstudio.LlmLoadModelConfigDict', return_value={}),
    mock.patch('transai.core.ai.saferandom.RandBits', return_value=5000),
    mock.patch('lmstudio.LlmPredictionConfigDict', return_value={}),
    mock.patch('lmstudio.Chat') as chat_cls,
  ):
    chat_cls.return_value = mock.MagicMock()
    worker.LoadModel(config)
    result: str = worker.ModelCall('my-lms-model', 'sys', 'user', str)
  assert result == 'e2e lms result'
