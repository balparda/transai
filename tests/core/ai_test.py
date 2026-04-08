# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""ai.py unittest.

Run with:
  poetry run pytest -vvv tests/core/ai_test.py
"""

from __future__ import annotations

import json
import math
import pathlib
from unittest import mock

import func_timeout.exceptions
import llama_cpp
import lmstudio
import pydantic
import pytest
from transcrypto.core import modmath
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

  def _LoadNew(self, _config: ai.AIModelConfig, /) -> ai.LoadedModel:
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
    _call_seed: int,
    /,
    *,
    images: list[ai.AIImageInput] | None = None,  # noqa: ARG002
    tools: list[ai.AnyCallable] | None = None,  # noqa: ARG002
  ) -> T:
    """Echo back the configured return value."""  # noqa: DOC201
    return self._call_return  # type: ignore[return-value]


def _MakeLlamaModel(config: ai.AIModelConfig | None = None) -> ai.LoadedModel:
  """Create a fake LoadedModel backed by a mock llama_cpp.Llama.

  Args:
    config: optional AIModelConfig to use for the model

  Returns:
    a LoadedModel dataclass with the given config (or a default valid one if None),
    fixed metadata, and a MagicMock spec'd as llama_cpp.Llama

  """
  if config is None:
    config = ai.MakeAIModelConfig(seed=5000)
  llm_mock = mock.MagicMock(spec=llama_cpp.Llama)
  return ai.LoadedModel(
    model_id=config['model_id'],
    seed_state=bytes(32),
    config=config,
    metadata={'key': 'val'},
    model=llm_mock,
  )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def testConstants() -> None:
  """Verify constants have sensible values."""
  assert 0.0 < ai.DEFAULT_GPU_RATIO <= 1.0
  assert 0.0 <= ai.DEFAULT_TEMPERATURE <= ai.MAX_TEMPERATURE
  assert modmath.IsPrime(ai.AI_MAX_SEED)


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
  config: ai.AIModelConfig = ai.MakeAIModelConfig(seed=5000)
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = ai.LoadedModel(
    model_id=ai.DEFAULT_TEXT_MODEL,
    seed_state=bytes(32),
    config=config,
    metadata={},
    model=llm_mock,
  )
  w.Close()
  llm_mock.close.assert_called_once()
  assert not w._loaded_models  # dict cleared


def testCloseDoesNotCallCloseOnLMStudioModels() -> None:
  """Close() must NOT call .close() on lmstudio.LLM instances (not in close-list)."""
  w = _ConcreteWorker()
  # Use a plain MagicMock instead of spec=lmstudio.LLM to avoid AttributeError
  # if lmstudio.LLM doesn't expose 'close' in its public interface
  lms_mock = mock.MagicMock()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(seed=5000)
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = ai.LoadedModel(
    model_id=ai.DEFAULT_TEXT_MODEL,
    seed_state=bytes(32),
    config=config,
    metadata={},
    model=lms_mock,
  )
  w.Close()
  # The mock has close() but the code should NOT call it since lmstudio.LLM
  # is not in _LLM_REQUIRING_CLOSE_METHOD
  lms_mock.close.assert_not_called()
  assert not w._loaded_models  # still cleared


def testCloseClearsModelsDict() -> None:
  """Close() must empty _loaded_models even when no models need close()."""
  w = _ConcreteWorker()
  lms_mock = mock.MagicMock(spec=lmstudio.LLM)
  config: ai.AIModelConfig = ai.MakeAIModelConfig(seed=5000)
  w._loaded_models['m1'] = ai.LoadedModel(
    model_id='m1',
    seed_state=bytes(32),
    config=config,
    metadata={},
    model=lms_mock,
  )
  w._loaded_models['m2'] = ai.LoadedModel(
    model_id='m2',
    seed_state=bytes(32),
    config=config,
    metadata={},
    model=lms_mock,
  )
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
  """_ConfigSeed must raise Error on negative seed (seed=-1 is reserved as 'no seed')."""
  w = _ConcreteWorker()
  with pytest.raises(ai.Error, match='seed'):
    w._ConfigSeed(ai.MakeAIModelConfig(seed=-2))


def testConfigSeedTreatsMinusOneSeedAsNone() -> None:
  """_ConfigSeed treats seed=-1 (reserved marker) as None and generates a random seed."""
  w = _ConcreteWorker()
  with mock.patch.object(saferandom, 'RandBits', return_value=12345) as rand_mock:
    result: ai.AIModelConfig = w._ConfigSeed(ai.MakeAIModelConfig(seed=-1))
  rand_mock.assert_called_once_with(31)
  assert result['seed'] == 12345


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
  """LoadModel() must return copies so mutations don't affect stored state."""
  w = _ConcreteWorker()
  loaded: ai.LoadedModel = _MakeLlamaModel()
  w._load_return = loaded
  cfg, _meta = w.LoadModel(ai.MakeAIModelConfig())
  stored_cfg: ai.AIModelConfig = w._loaded_models[ai.DEFAULT_TEXT_MODEL].config
  # Returned config is a copy (not the same object) but equal in content
  assert cfg == stored_cfg
  assert cfg is not stored_cfg


def testLoadModelForcesReloadWhenSeedSpecified() -> None:
  """LoadModel sets force=True when config has an explicit seed (lines 219-220)."""
  w = _ConcreteWorker()
  w._load_return = _MakeLlamaModel()
  # Explicit non-None seed triggers force=True at the start of LoadModel
  cfg, _ = w.LoadModel(ai.MakeAIModelConfig(seed=1000))
  assert cfg is not None


def testLoadModelRaisesWhenConfigSeedReturnsBadSeed() -> None:
  """LoadModel raises Error when _ConfigSeed returns a config with bad seed (safety guard)."""
  w = _ConcreteWorker()
  bad_config: ai.AIModelConfig = ai.MakeAIModelConfig(seed=5000)
  bad_config.update({'seed': 0})  # type: ignore[typeddict-item]
  with (
    mock.patch.object(w, '_ConfigSeed', return_value=bad_config),
    pytest.raises(ai.Error, match='seed to be loaded'),
  ):
    w.LoadModel(ai.MakeAIModelConfig())


def testLoadModelReturnsCachedForExistingModel() -> None:
  """LoadModel returns cached (config, metadata) copies on repeated calls for same model."""
  w = _ConcreteWorker()
  # Use an explicit lowercase model_id so the _ConfigSeed normalization matches the stored key
  model_config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='test-model', seed=5000)
  loaded = ai.LoadedModel(
    model_id='test-model',
    seed_state=bytes(32),
    config=model_config,
    metadata={'info': 'val'},
    model=mock.MagicMock(spec=llama_cpp.Llama),
  )
  w._load_return = loaded
  # First load stores the model
  cfg1, meta1 = w.LoadModel(ai.MakeAIModelConfig(model_id='test-model'))
  # Second load must return cached, not call _LoadNew again
  w._load_return = None  # _LoadNew would raise if called
  cfg2, meta2 = w.LoadModel(ai.MakeAIModelConfig(model_id='test-model'))
  assert cfg1 == cfg2
  assert meta1 == meta2


def testLoadModelReturnsCachedForQuantizedVariant() -> None:
  """LoadModel returns cached base model when a quantized variant is requested."""
  w = _ConcreteWorker()
  # Load the base model (no quant suffix)
  base_config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='my-model', seed=5000)
  loaded = ai.LoadedModel(
    model_id='my-model',
    seed_state=bytes(32),
    config=base_config,
    metadata={},
    model=mock.MagicMock(spec=llama_cpp.Llama),
  )
  w._load_return = loaded
  w.LoadModel(ai.MakeAIModelConfig(model_id='my-model'))
  # Requesting a quantized variant should find the already-loaded base via ignore_quant
  w._load_return = None  # _LoadNew would raise if called
  cfg, _ = w.LoadModel(ai.MakeAIModelConfig(model_id='my-model@q8_0'))
  assert cfg['model_id'] == 'my-model'


def testLoadModelWrapsNonErrorExceptionFromLoadNew() -> None:
  """LoadModel() must catch generic exceptions from _LoadNew() and re-raise as Error."""
  w = _ConcreteWorker()
  with (
    mock.patch.object(w, '_LoadNew', side_effect=RuntimeError('backend exploded')),
    pytest.raises(ai.Error, match='load error'),
  ):
    w.LoadModel(ai.MakeAIModelConfig())


# ---------------------------------------------------------------------------
# AIWorker._RegisterModel
# ---------------------------------------------------------------------------


def testRegisterModelStoresModel() -> None:
  """_RegisterModel stores a LoadedModel in _loaded_models with normalized model_id."""
  w = _ConcreteWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(model_id='reg-model', seed=5000)
  model = ai.LoadedModel(
    model_id='reg-model',
    seed_state=bytes(32),
    config=config,
    metadata={'k': 'v'},
    model=mock.MagicMock(spec=llama_cpp.Llama),
  )
  w._RegisterModel(model)
  assert 'reg-model' in w._loaded_models
  assert w._loaded_models['reg-model'].config['model_id'] == 'reg-model'
  assert w._loaded_models['reg-model'].metadata == {'k': 'v'}


def testRegisterModelRaisesOnBadSeed() -> None:
  """_RegisterModel raises Error when _ConfigSeed returns a config with bad seed (safety guard)."""
  w = _ConcreteWorker()
  model: ai.LoadedModel = _MakeLlamaModel()
  bad_config: ai.AIModelConfig = ai.MakeAIModelConfig(seed=5000)
  bad_config.update({'seed': 0})  # type: ignore[typeddict-item]
  with (
    mock.patch.object(w, '_ConfigSeed', return_value=bad_config),
    pytest.raises(ai.Error, match='seed to be registered'),
  ):
    w._RegisterModel(model)


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


def testModelCallRaisesWhenImagesButNoVisionCapability() -> None:
  """ModelCall() raises Error when images are provided but model has vision=False."""
  w = _ConcreteWorker()
  loaded: ai.LoadedModel = _MakeLlamaModel(ai.MakeAIModelConfig(vision=False, seed=5000))
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = loaded
  with pytest.raises(ai.Error, match='does not support vision inputs'):
    w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str, images=[b'\x89PNG'])


def testModelCallWrapsJsonDecodeErrorFromCall() -> None:
  """ModelCall() must catch json.JSONDecodeError from _Call() and re-raise as Error."""
  w = _ConcreteWorker()
  loaded: ai.LoadedModel = _MakeLlamaModel()
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = loaded
  with (
    mock.patch.object(w, '_Call', side_effect=json.JSONDecodeError('bad json', '', 0)),
    pytest.raises(ai.Error, match='invalid JSON output'),
  ):
    w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str)


def testModelCallWrapsGenericExceptionFromCall() -> None:
  """ModelCall() must catch generic exceptions from _Call() and re-raise as Error."""
  w = _ConcreteWorker()
  loaded: ai.LoadedModel = _MakeLlamaModel()
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = loaded
  with (
    mock.patch.object(w, '_Call', side_effect=RuntimeError('backend exploded')),
    pytest.raises(ai.Error, match='call error'),
  ):
    w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str)


def testModelCallPassesImagesThrough() -> None:
  """ModelCall() must forward images keyword argument to _Call()."""
  w = _ConcreteWorker()
  w._call_return = 'img-result'
  loaded: ai.LoadedModel = _MakeLlamaModel(ai.MakeAIModelConfig(vision=True, seed=5000))
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
  loaded: ai.LoadedModel = _MakeLlamaModel(ai.MakeAIModelConfig(vision=True, seed=5000))
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


def testModelCallRaisesWhenToolsButNoToolingCapability() -> None:
  """ModelCall() raises Error when tools provided but model has tooling=False."""
  w = _ConcreteWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(tooling=False, seed=5000)
  loaded = ai.LoadedModel(
    model_id=ai.DEFAULT_TEXT_MODEL,
    seed_state=bytes(32),
    config=config,
    metadata={},
    model=mock.MagicMock(spec=llama_cpp.Llama),
  )
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = loaded
  with pytest.raises(ai.Error, match='not trained for tool use'):
    w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str, tools=['math.gcd'])


def testModelCallRaisesWhenToolsAndStructuredOutput() -> None:
  """ModelCall() raises Error when tools requested but output_format is not str."""

  class _MyOutput(pydantic.BaseModel):
    value: int

  w = _ConcreteWorker()
  config: ai.AIModelConfig = ai.MakeAIModelConfig(tooling=True, seed=5000)
  loaded = ai.LoadedModel(
    model_id=ai.DEFAULT_TEXT_MODEL,
    seed_state=bytes(32),
    config=config,
    metadata={},
    model=mock.MagicMock(spec=llama_cpp.Llama),
  )
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = loaded
  with pytest.raises(ai.Error, match='non-str output'):
    w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', _MyOutput, tools=['math.gcd'])


def testModelCallPassesToolsThroughToCall() -> None:
  """ModelCall() converts tool strings to callables and passes them to _Call()."""

  def _my_tool(x: int) -> int:
    """Return double the input.

    Args:
      x: the integer to double

    Returns:
      twice x

    """
    return x * 2

  w = _ConcreteWorker()
  w._call_return = 'tool result'
  config: ai.AIModelConfig = ai.MakeAIModelConfig(tooling=True, seed=5000)
  loaded = ai.LoadedModel(
    model_id=ai.DEFAULT_TEXT_MODEL,
    seed_state=bytes(32),
    config=config,
    metadata={},
    model=mock.MagicMock(spec=llama_cpp.Llama),
  )
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = loaded
  with mock.patch.object(w, '_Call', return_value='tool result') as call_mock:
    result: str = w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str, tools=[_my_tool])
  assert result == 'tool result'
  call_mock.assert_called_once()
  _, kwargs = call_mock.call_args
  assert kwargs.get('tools') == [_my_tool]


# ---------------------------------------------------------------------------
# _GetCallable
# ---------------------------------------------------------------------------


def testGetCallableWithCallable() -> None:
  """_GetCallable returns a callable object unchanged."""
  func = lambda x: x  # pyright: ignore[reportUnknownVariableType, reportUnknownLambdaType]
  assert ai._GetCallable(func) is func  # pyright: ignore[reportUnknownArgumentType]


def testGetCallableWithValidDotString() -> None:
  """_GetCallable resolves a fully qualified name string to the callable."""
  result = ai._GetCallable('math.gcd')
  assert result is math.gcd


def testGetCallableRaisesOnImportError() -> None:
  """_GetCallable raises Error when the module cannot be imported."""
  with pytest.raises(ai.Error, match='Error resolving tool name'):
    ai._GetCallable('nonexistent_xyz_module_abc.some_func')


def testGetCallableRaisesOnAttributeError() -> None:
  """_GetCallable raises Error when the attribute does not exist in the module."""
  with pytest.raises(ai.Error, match='Error resolving tool name'):
    ai._GetCallable('math.nonexistent_attr_xyz')


def testGetCallableRaisesOnNonCallable() -> None:
  """_GetCallable raises Error when the resolved symbol is not callable."""
  # os.sep is a string (path separator), not callable
  with pytest.raises(ai.Error, match='not callable'):
    ai._GetCallable('os.sep')


# ---------------------------------------------------------------------------
# AIWorker timeout (func_timeout integration)
# ---------------------------------------------------------------------------


def testLoadModelNoTimeoutCallsFuncDirectly() -> None:
  """LoadModel with timeout=None calls _LoadNew directly without using func_timeout."""
  w = _ConcreteWorker()
  w._timeout = None
  w._load_return = _MakeLlamaModel()
  with mock.patch('transai.core.ai.func_timeout.func_timeout') as ft_mock:
    w.LoadModel(ai.MakeAIModelConfig())
  ft_mock.assert_not_called()


def testModelCallNoTimeoutCallsFuncDirectly() -> None:
  """ModelCall with timeout=None calls _Call directly without using func_timeout."""
  w = _ConcreteWorker()
  w._timeout = None
  w._call_return = 'answer'
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = _MakeLlamaModel()
  with mock.patch('transai.core.ai.func_timeout.func_timeout') as ft_mock:
    result: str = w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str)
  assert result == 'answer'
  ft_mock.assert_not_called()


def testLoadModelRaisesOnTimeout() -> None:
  """LoadModel() wraps FunctionTimedOut from func_timeout as ai.Error."""
  w = _ConcreteWorker()
  w._timeout = 1.0
  timed_out = func_timeout.exceptions.FunctionTimedOut()
  with (
    mock.patch('transai.core.ai.func_timeout.func_timeout', side_effect=timed_out),
    pytest.raises(ai.Error, match='load timed out'),
  ):
    w.LoadModel(ai.MakeAIModelConfig())


def testModelCallRaisesOnTimeout() -> None:
  """ModelCall() wraps FunctionTimedOut from func_timeout as ai.Error."""
  w = _ConcreteWorker()
  w._timeout = 1.0
  w._loaded_models[ai.DEFAULT_TEXT_MODEL] = _MakeLlamaModel()
  timed_out = func_timeout.exceptions.FunctionTimedOut()
  with (
    mock.patch('transai.core.ai.func_timeout.func_timeout', side_effect=timed_out),
    pytest.raises(ai.Error, match='call timed out'),
  ):
    w.ModelCall(ai.DEFAULT_TEXT_MODEL, 'sys', 'user', str)
