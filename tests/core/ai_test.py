# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""ai.py unittest.

Run with:
  poetry run pytest -vvv tests/core/ai_test.py
"""

from __future__ import annotations

import pathlib
from unittest import mock

import llama_cpp
import lmstudio
import pydantic
import pytest
from transcrypto.utils import saferandom

from transai.core import ai

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConcreteWorker(ai.AIWorker):
  """Minimal concrete AIWorker used only in tests."""

  def __init__(self) -> None:
    """Init."""
    super().__init__()
    self._load_return: ai.LoadedModel | None = None
    self._call_return: object = 'default'

  def _Load(self, _config: ai.AIModelConfig, /) -> ai.LoadedModel:
    """Return whatever was configured for the test."""  # noqa: DOC201, DOC501
    if self._load_return is None:
      raise ai.Error('_load_return not set')
    return self._load_return

  def _Call[T: pydantic.BaseModel | str](
    self,
    _model: ai.LoadedModel,
    _system_prompt: str,
    _user_prompt: str,
    _output_format: type[T],
    /,
    *,
    images: list[ai.AIImageInput] | None = None,  # noqa: ARG002
  ) -> T:
    """Echo back the configured return value."""  # noqa: DOC201
    return self._call_return  # type: ignore[return-value]


def _MakeLlamaModel(config: ai.AIModelConfig | None = None) -> ai.LoadedModel:
  """Create a fake LoadedModel backed by a mock llama_cpp.Llama.

  Args:
    config: optional AIModelConfig to use for the model

  Returns:
    a LoadedModel tuple with the given config (or a default valid one if None),
    empty metadata, and a MagicMock spec'd as llama_cpp.Llama

  """
  if config is None:
    config = ai.MakeAIModelConfig()
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  return (config, {'key': 'val'}, llm_mock)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def testConstants() -> None:
  """Verify constants have sensible values."""
  assert 0.0 < ai.DEFAULT_GPU_RATIO <= 1.0
  assert 0.0 <= ai.DEFAULT_TEMPERATURE <= ai.MAX_TEMPERATURE


# ---------------------------------------------------------------------------
# AIWorker context manager
# ---------------------------------------------------------------------------


def testContextManagerEnterReturnsSelf() -> None:
  """__enter__ must return self."""
  w = _ConcreteWorker()
  assert w.__enter__() is w  # noqa: PLC2801


def testContextManagerExitCallsClose() -> None:
  """__exit__ must delegate to Close()."""
  w = _ConcreteWorker()
  with mock.patch.object(w, 'Close') as close_mock:
    w.__exit__(None, None, None)
    close_mock.assert_called_once()


def testContextManagerUsedAsContextManager() -> None:
  """Using `with` syntax calls __enter__ and __exit__."""
  w = _ConcreteWorker()
  with mock.patch.object(w, 'Close') as close_mock:
    with w:
      pass
    close_mock.assert_called_once()


# ---------------------------------------------------------------------------
# AIWorker.Close
# ---------------------------------------------------------------------------


def testCloseCallsCloseOnLlamaModels() -> None:
  """Close() must call .close() on llama_cpp.Llama instances."""
  w = _ConcreteWorker()
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = (config, {}, llm_mock)
  w.Close()
  llm_mock.close.assert_called_once()
  assert not w._loaded_models  # dict cleared


def testCloseDoesNotCallCloseOnLMStudioModels() -> None:
  """Close() must NOT call .close() on lmstudio.LLM instances (not in close-list)."""
  w = _ConcreteWorker()
  # Use a plain MagicMock instead of spec=lmstudio.LLM to avoid AttributeError
  # if lmstudio.LLM doesn't expose 'close' in its public interface
  lms_mock = mock.MagicMock()
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = (config, {}, lms_mock)
  w.Close()
  # The mock has close() but the code should NOT call it since lmstudio.LLM
  # is not in _LLM_REQUIRING_CLOSE_METHOD
  lms_mock.close.assert_not_called()
  assert not w._loaded_models  # still cleared


def testCloseClearsModelsDict() -> None:
  """Close() must empty _loaded_models even when no models need close()."""
  w = _ConcreteWorker()
  lms_mock = mock.MagicMock(spec=lmstudio.LLM)
  config: ai.AIModelConfig = ai.MakeAIModelConfig()
  w._loaded_models['m1'] = (config, {}, lms_mock)
  w._loaded_models['m2'] = (config, {}, lms_mock)
  w.Close()
  assert w._loaded_models == {}


# ---------------------------------------------------------------------------
# AIWorker._ConfigSeed
# ---------------------------------------------------------------------------


def testConfigSeedStripsAndLowercasesModelId() -> None:
  """_ConfigSeed must normalize model_id."""
  w = _ConcreteWorker()
  result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(model_id='  MyModel  '))
  assert result['model_id'] == 'mymodel'


def testConfigSeedInjectsRandomSeedWhenNone() -> None:
  """_ConfigSeed must inject a random seed when seed=None."""
  w = _ConcreteWorker()
  with mock.patch.object(saferandom, 'RandBits', return_value=12345) as rand_mock:
    result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(seed=None))
  rand_mock.assert_called_once_with(31)
  assert result['seed'] == 12345


def testConfigSeedKeepsExplicitSeed() -> None:
  """Explicit seed is kept, RandBits should NOT be called."""
  w = _ConcreteWorker()
  with mock.patch.object(saferandom, 'RandBits') as rand_mock:
    result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(seed=99))
  rand_mock.assert_not_called()
  assert result['seed'] == 99


def testConfigSeedRaisesOnEmptyModelId() -> None:
  """_ConfigSeed must raise Error on empty model_id."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='model_id'):
    w._ConfigSeed(ai.MakeAIModelConfig(model_id='   '))


def testConfigSeedRaisesOnContextTooSmall() -> None:
  """_ConfigSeed must raise Error when context < 16."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='context'):
    w._ConfigSeed(ai.MakeAIModelConfig(context=8))


def testConfigSeedRaisesOnContextTooLarge() -> None:
  """_ConfigSeed must raise Error when context > AI_MAX_CONTEXT."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='context'):
    w._ConfigSeed(ai.MakeAIModelConfig(context=ai.AI_MAX_CONTEXT + 1))


def testConfigSeedRaisesOnTemperatureTooLow() -> None:
  """_ConfigSeed must raise Error on negative temperature."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='temperature'):
    w._ConfigSeed(ai.MakeAIModelConfig(temperature=-0.1))


def testConfigSeedRaisesOnTemperatureTooHigh() -> None:
  """_ConfigSeed must raise Error when temperature > MAX_TEMPERATURE."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='temperature'):
    w._ConfigSeed(ai.MakeAIModelConfig(temperature=ai.MAX_TEMPERATURE + 0.01))


def testConfigSeedRaisesOnGpuRatioTooLow() -> None:
  """_ConfigSeed must raise Error when gpu_ratio < 0.1."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='gpu_ratio'):
    w._ConfigSeed(ai.MakeAIModelConfig(gpu_ratio=0.05))


def testConfigSeedRaisesOnGpuRatioTooHigh() -> None:
  """_ConfigSeed must raise Error when gpu_ratio > 1.0."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='gpu_ratio'):
    w._ConfigSeed(ai.MakeAIModelConfig(gpu_ratio=1.01))


def testConfigSeedRaisesOnSeedOutOfRange() -> None:
  """_ConfigSeed must raise Error when explicit seed is out of valid range."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='seed'):
    w._ConfigSeed(ai.MakeAIModelConfig(seed=0))


def testConfigSeedRaisesOnSeedNegative() -> None:
  """_ConfigSeed must raise Error on negative seed."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='seed'):
    w._ConfigSeed(ai.MakeAIModelConfig(seed=-1))


def testConfigSeedRaisesOnSeedTooLarge() -> None:
  """_ConfigSeed must raise Error when seed > AI_MAX_SEED."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='seed'):
    w._ConfigSeed(ai.MakeAIModelConfig(seed=ai.AI_MAX_SEED + 1))


def testConfigSeedAcceptsMinContext() -> None:
  """Boundary: context=16 is valid."""
  w = _ConcreteWorker()
  result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(context=16))
  assert result['context'] == 16


def testConfigSeedAcceptsMaxContext() -> None:
  """Boundary: context=AI_MAX_CONTEXT is valid."""
  w = _ConcreteWorker()
  result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(context=ai.AI_MAX_CONTEXT))
  assert result['context'] == ai.AI_MAX_CONTEXT


def testConfigSeedAcceptsZeroTemperature() -> None:
  """Boundary: temperature=0.0 is valid."""
  w = _ConcreteWorker()
  result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(temperature=0.0))
  assert result['temperature'] == pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]


def testConfigSeedAcceptsMaxTemperature() -> None:
  """Boundary: temperature=MAX_TEMPERATURE is valid."""
  w = _ConcreteWorker()
  result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(temperature=ai.MAX_TEMPERATURE))
  assert result['temperature'] == pytest.approx(ai.MAX_TEMPERATURE)  # pyright: ignore[reportUnknownMemberType]


def testConfigSeedAcceptsMinGpuRatio() -> None:
  """Boundary: gpu_ratio=0.1 is valid."""
  w = _ConcreteWorker()
  result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(gpu_ratio=0.1))
  assert result['gpu_ratio'] == pytest.approx(0.1)  # pyright: ignore[reportUnknownMemberType]


def testConfigSeedAcceptsMaxGpuRatio() -> None:
  """Boundary: gpu_ratio=1.0 is valid."""
  w = _ConcreteWorker()
  result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(gpu_ratio=1.0))
  assert result['gpu_ratio'] == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# AIWorker.LoadModel
# ---------------------------------------------------------------------------


def testLoadModelStoresAndReturnsConfig() -> None:
  """LoadModel() must store the model and return (config, metadata)."""
  w = _ConcreteWorker()
  loaded: ai.LoadedModel = _MakeLlamaModel()
  w._load_return = loaded
  cfg, meta = w.LoadModel(ai.MakeAIModelConfig())
  assert cfg['model_id'] == ai.DEFAULT_TEXT_MODEL
  assert meta == {'key': 'val'}
  assert ai.DEFAULT_TEXT_MODEL in w._loaded_models


def testLoadModelStoresIsolatedCopies() -> None:
  """LoadModel() must store copies so mutations of the original don't affect stored state."""
  w = _ConcreteWorker()
  loaded: ai.LoadedModel = _MakeLlamaModel()
  w._load_return = loaded
  cfg, _meta = w.LoadModel(ai.MakeAIModelConfig())
  stored_cfg: ai.AIModelConfig = w._loaded_models[ai.DEFAULT_TEXT_MODEL][0]
  # They should be equal content but distinct objects
  assert cfg == stored_cfg
  assert cfg is not stored_cfg


# ---------------------------------------------------------------------------
# AIWorker.ModelCall
# ---------------------------------------------------------------------------


def testModelCallDelegatesToCall() -> None:
  """ModelCall() must delegate to _Call() for a known model."""
  w = _ConcreteWorker()
  w._call_return = 'hello'
  loaded: ai.LoadedModel = _MakeLlamaModel()
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = loaded
  result: str = w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str)
  assert result == 'hello'


def testModelCallRaisesForUnknownModel() -> None:
  """ModelCall() must raise Error for a model that was not loaded."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='not loaded'):
    w.ModelCall('unknown', 'sys', 'user', str)


def testModelCallPassesImagesThrough() -> None:
  """ModelCall() must forward images keyword argument to _Call()."""
  w = _ConcreteWorker()
  w._call_return = 'img-result'
  loaded: ai.LoadedModel = _MakeLlamaModel()
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = loaded
  images: list[ai.AIImageInput] = [b'\x89PNG']
  # patch _Call to capture invocation
  with mock.patch.object(w, '_Call', return_value='img-result') as call_mock:
    w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str, images=images)
  call_mock.assert_called_once()
  _, kwargs = call_mock.call_args
  assert kwargs.get('images') == images


def testModelCallPassesMixedImageTypesThrough() -> None:
  """ModelCall() must forward a mixed list of bytes, Path, and str images to _Call()."""
  w = _ConcreteWorker()
  w._call_return = 'mixed-result'
  loaded: ai.LoadedModel = _MakeLlamaModel()
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = loaded
  mixed_images: list[ai.AIImageInput] = [
    b'\x89PNG\r\n\x1a\n',  # bytes
    pathlib.Path('/some/image.png'),  # pathlib.Path
    '/another/image.jpg',  # str path
  ]
  with mock.patch.object(w, '_Call', return_value='mixed-result') as call_mock:
    result: str = w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str, images=mixed_images)
  assert result == 'mixed-result'
  call_mock.assert_called_once()
  _, kwargs = call_mock.call_args
  forwarded = kwargs.get('images')
  assert forwarded is mixed_images
  assert isinstance(forwarded[0], bytes)
  assert isinstance(forwarded[1], pathlib.Path)
  assert isinstance(forwarded[2], str)
