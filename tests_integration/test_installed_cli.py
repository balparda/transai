# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: build wheel, install into a fresh venv, run the installed CLI.

Why this exists (vs normal unit tests):
- Unit tests (CliRunner) validate CLI wiring while running from the source tree.
- This test validates *packaging*: the wheel builds, installs, and the console script works.

What we verify:
- ``transai --version`` prints the expected version.
- ``transai --no-lms query ...`` with a real tiny GGUF model produces text output.

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

import huggingface_hub
import pytest
from transcrypto.utils import base, config

import transai

_APP_NAME: str = 'transai'  # this is the directory name, the package name
_APP_NAMES: set[str] = {'transai'}  # this is the console scripts names

# ggml-org/tinygemma3-GGUF: official llama.cpp CI test model:
# <https://huggingface.co/ggml-org/tinygemma3-GGUF>
_HF_REPO: str = 'ggml-org/tinygemma3-GGUF'
_GGUF_FILE: str = 'tinygemma3-Q8_0.gguf'
_MMPROJ_FILE: str = 'mmproj-tinygemma3.gguf'
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
  # TODO: cached_mmproj: str = huggingface_hub.hf_hub_download(_HF_REPO, _MMPROJ_FILE)
  root: pathlib.Path = tmp_path_factory.mktemp('models')
  model_dir: pathlib.Path = root / _MODEL_ID
  model_dir.mkdir()
  (model_dir / _GGUF_FILE).symlink_to(cached_gguf)
  # TODO: do the same (model_dir / _MMPROJ_FILE).symlink_to(cached_mmproj)
  return root


@pytest.mark.integration
@pytest.mark.slow
def test_installed_cli_smoke(tmp_path: pathlib.Path, models_root: pathlib.Path) -> None:
  """Build wheel, install into a clean venv, run the installed CLIs."""
  repo_root: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
  expected_version: str = transai.__version__
  vpy, bin_dir = config.EnsureAndInstallWheel(repo_root, tmp_path, expected_version, _APP_NAMES)
  cli_paths: dict[str, pathlib.Path] = config.EnsureConsoleScriptsPrintExpectedVersion(
    vpy, bin_dir, expected_version, _APP_NAMES
  )
  # basic command smoke tests
  data_dir: pathlib.Path = config.CallGetConfigDirFromVEnv(vpy, _APP_NAME)
  _version_call(cli_paths)
  _query_call(cli_paths, models_root, data_dir)


def _version_call(cli_paths: dict[str, pathlib.Path], /) -> None:
  """Verify ``--version`` prints the expected version string."""
  r = base.Run([str(cli_paths['transai']), '--version'])
  assert '1.' in r.stdout
  assert '\x1b[' not in r.stdout  # no ANSI escape sequences
  assert '\x1b[' not in r.stderr


def _query_call(
  cli_paths: dict[str, pathlib.Path], models_root: pathlib.Path, data_dir: pathlib.Path, /
) -> None:
  """Run a real llama.cpp text query with the tiny test model.

  Uses ``--no-lms`` to bypass LM Studio, ``-s 999`` for a deterministic
  reproducible seed, ``--no-flash`` and ``--context 512`` to keep the test
  fast and portable.

  Args:
    cli_paths: map of console-script name -> installed path
    models_root: temporary directory containing the model sub-folder
    data_dir: transai config directory (cleaned up on exit)

  """
  try:
    r = base.Run(
      [
        str(cli_paths['transai']),
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
        'capital of france',
      ]
    )
    # The tiny random model produces garbage, but we verify the full CLI
    # pipeline works end-to-end: model download ➜ load ➜ inference ➜ print.
    assert r.stdout, 'Expected non-empty stdout from model query'
    assert '的管理' in r.stdout and 'What is this:' in r.stdout
    assert '\x1b[' not in r.stdout  # no ANSI escape sequences
    assert '\x1b[' not in r.stderr
  finally:
    if data_dir.exists():
      shutil.rmtree(data_dir)  # remove created data to isolate the next CLI's read step
