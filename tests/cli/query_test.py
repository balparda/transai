# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: query.py."""

from __future__ import annotations

import pathlib
from unittest import mock

import pytest
from click import testing
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging

from tests import transai_test
from transai.core import ai, llama, lms


@pytest.fixture(autouse=True)
def reset_cli() -> None:
  """Reset CLI singleton before each test."""
  cli_logging.ResetConsole()
  app_config.ResetConfig()


def testQueryRaisesErrorWhenNoLMSAndNoRoot() -> None:
  """Query command handles ai.Error when lms=False and models_root=None.

  We patch TransAIConfig so ctx.obj.lms=False and ctx.obj.models_root=None,
  forcing the guard condition to trigger and raise ai.Error, which CLIErrorGuard
  catches and routes to obj.console.print().

  """
  fake_config = mock.MagicMock()
  fake_config.lms = False
  fake_config.models_root = None
  fake_config.verbose = 0
  with (
    mock.patch('transai.transai.TransAIConfig', return_value=fake_config),
    mock.patch.object(lms, 'LMStudioWorker') as lms_mock,
    mock.patch.object(llama, 'LlamaWorker') as llama_mock,
  ):
    result: testing.Result = transai_test.CallCLI(['--no-lms', 'query', 'Make me a recipe.'])
  assert result.exit_code == 0  # CLIErrorGuard catches error and returns normally
  lms_mock.assert_not_called()  # neither worker should have been created
  llama_mock.assert_not_called()
  fake_config.console.print.assert_called_with(
    'Non-LM Studio client library requires `models_root` to be set'
  )


def testQueryUsesLMStudioWorker() -> None:
  """Query command uses LMStudioWorker when --lms (the default)."""
  worker_mock = mock.MagicMock()
  worker_mock.LoadModel.return_value = (ai.MakeAIModelConfig(), {})
  worker_mock.ModelCall.return_value = 'Paris'
  with mock.patch.object(lms, 'LMStudioWorker', return_value=worker_mock):
    result: testing.Result = transai_test.CallCLI(['query', 'What is the capital of France?'])
  assert result.exit_code == 0, result.output
  worker_mock.LoadModel.assert_called_once()
  worker_mock.ModelCall.assert_called_once()


def testQueryUsesLlamaWorkerWhenNoLMS(tmp_path: pathlib.Path) -> None:
  """Query command uses LlamaWorker when --no-lms and --root are provided."""
  worker_mock = mock.MagicMock()
  worker_mock.LoadModel.return_value = (ai.MakeAIModelConfig(), {})
  worker_mock.ModelCall.return_value = 'Bonjour'
  with mock.patch.object(llama, 'LlamaWorker', return_value=worker_mock):
    result: testing.Result = transai_test.CallCLI(
      ['--no-lms', '--root', str(tmp_path), 'query', 'hi']
    )
  assert result.exit_code == 0, result.output
  worker_mock.LoadModel.assert_called_once()
  worker_mock.ModelCall.assert_called_once()


def testQueryWarnsSeedWithNoFreeResources() -> None:
  """Query logs a warning when seed is set and free_resources=False (default, line 58)."""
  worker_mock = mock.MagicMock()
  worker_mock.LoadModel.return_value = (ai.MakeAIModelConfig(), {})
  worker_mock.ModelCall.return_value = 'seeded answer'
  with mock.patch.object(lms, 'LMStudioWorker', return_value=worker_mock):
    # --seed 5000 sets config.seed=5000; free_resources defaults to False → warning fires
    result: testing.Result = transai_test.CallCLI(['--seed', '5000', 'query', 'hello'])
  assert result.exit_code == 0, result.output
  worker_mock.LoadModel.assert_called_once()
  worker_mock.ModelCall.assert_called_once()
