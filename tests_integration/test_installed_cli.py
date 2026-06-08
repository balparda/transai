# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda
# SPDX-License-Identifier: Apache-2.0
"""Integration tests: build wheel, install into a fresh venv, run the installed CLI.

The test model is ``ggml-org/tinygemma3-GGUF`` (~47 MB, 38M params, random weights,
WTFPL license) — the official llama.cpp CI test model.  It produces garbage output
(random weights) but is fast to download and fully compatible with llama-cpp-python.

Vision smoke-test is skipped for now: llama-cpp-python has no Gemma-3 vision
chat-handler, so the mmproj cannot be exercised without crashing.  When
``llama-cpp-python`` ships a native Gemma-3 handler, add the mmproj file to
``_models_root`` and extend ``_query_call`` with an ``-i`` image argument.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess  # noqa: S404

import huggingface_hub
import pytest
from transcrypto.utils import base

import transai

# ggml-org/tinygemma3-GGUF: official llama.cpp CI test model:
# <https://huggingface.co/ggml-org/tinygemma3-GGUF>
_HF_REPO: str = 'ggml-org/tinygemma3-GGUF'
_GGUF_FILE: str = 'tinygemma3-Q8_0.gguf'  # 47 MB
_MMPROJ_FILE: str = 'mmproj-tinygemma3.gguf'  # 1 MB
_MODEL_ID: str = 'tinygemma3'


@pytest.fixture(scope='module')
def models_root(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
  """Download the tiny test GGUF into a temporary models-root directory.

  The directory layout mirrors what ``transai`` expects::

      <root>/<model-id>/<model>.gguf

  The GGUF is symlinked from the Hugging Face cache to avoid duplication.

  Args:
    tmp_path_factory: pytest factory for module-scoped temp paths

  Returns:
    pathlib.Path: the root directory containing the model subdirectory

  """
  cached_gguf: str = huggingface_hub.hf_hub_download(_HF_REPO, _GGUF_FILE)  # pyright: ignore[reportUnknownMemberType]
  root: pathlib.Path = tmp_path_factory.mktemp('models')
  model_dir: pathlib.Path = root / _MODEL_ID
  model_dir.mkdir()
  (model_dir / _GGUF_FILE).symlink_to(cached_gguf)
  return root


@pytest.mark.integration
@pytest.mark.slow
def test_installed_cli_smoke(models_root: pathlib.Path) -> None:
  """Test the installed CLI from the current environment."""
  # find the installed console script; will raise if not found
  cli_path: str | None = shutil.which('transai')
  if cli_path is None:
    pytest.fail('Console script "transai" not found in PATH')
  cli: pathlib.Path = pathlib.Path(cli_path)
  # verify version
  base.VersionCallCheck(cli, transai.__version__)
  # basic command smoke tests
  _query_call(cli, models_root)


def _query_call(cli: pathlib.Path, models_root: pathlib.Path) -> None:
  """Run a real llama.cpp text query with the tiny test model.

  Uses ``--no-lms`` to bypass LM Studio, ``-s 999`` for a deterministic
  reproducible seed, ``--no-flash`` and ``--context 512`` to keep the test
  fast and portable.

  """
  r: subprocess.CompletedProcess[str] = base.Run(
    # run
    [
      str(cli),
      '--no-lms',
      '-m',
      _MODEL_ID,
      '-s',
      '999',
      '--no-flash',
      '--context',
      '512',
      '--gpu-layers',
      '0',
      '-r',
      str(models_root),
      '--no-color',
      'query',
      '"capital of france"',
      '--free',
    ]
  )
  # The tiny random model produces garbage, but we verify the full CLI
  # pipeline works end-to-end: model download ➜ load ➜ inference ➜ print.
  assert r.stdout, 'Expected non-empty stdout from model query'
  assert '\x1b[' not in r.stdout  # no ANSI escape sequences
  assert '\x1b[' not in r.stderr
