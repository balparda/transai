# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda
# SPDX-License-Identifier: Apache-2.0
"""Integration tests: build wheel, install into a fresh venv, run the installed CLI.

The `Qwen3-VL-2B` model we use is a functional vision model, but even so it is 2+ GB
and so is excluded from Github runs.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess  # noqa: S404

import huggingface_hub
import pytest

# ggml-org/Qwen3-VL-2B-Instruct-GGUF: functional vision model 2+ GB:
# <https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF>
_HF_REPO: str = 'ggml-org/Qwen3-VL-2B-Instruct-GGUF'
_GGUF_FILE: str = 'Qwen3-VL-2B-Instruct-Q8_0.gguf'  # 1.83 GB
_MMPROJ_FILE: str = 'mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf'  # 445 MB
_MODEL_ID: str = 'Qwen3-VL-2B'

# On macOS, llama.cpp's Metal backend sometimes fires a SIGABRT (exit -6) during
# GPU resource cleanup in __cxa_finalize_ranges / atexit, *after* the model has
# already written its full output to stdout.  The inference itself succeeded; we
# treat this platform-specific cleanup crash as non-fatal.
_MACOS_METAL_CLEANUP_EXIT: int = -6  # SIGABRT

_TEST_IMAGES_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / 'tests' / 'data' / 'images'
_IMG_100: pathlib.Path = _TEST_IMAGES_PATH / '100.jpg'  # Bach
_IMG_107: pathlib.Path = _TEST_IMAGES_PATH / '107.png'  # Schubert


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
  cached_mmproj: str = huggingface_hub.hf_hub_download(_HF_REPO, _MMPROJ_FILE)  # pyright: ignore[reportUnknownMemberType]
  root: pathlib.Path = tmp_path_factory.mktemp('models')
  model_dir: pathlib.Path = root / _MODEL_ID
  model_dir.mkdir()
  (model_dir / _GGUF_FILE).symlink_to(cached_gguf)
  (model_dir / _MMPROJ_FILE).symlink_to(cached_mmproj)
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
  _query_call(cli, models_root)


def _query_call(cli: pathlib.Path, models_root: pathlib.Path) -> None:
  """Run a real llama.cpp text query with the tiny test model.

  Uses ``--no-lms`` to bypass LM Studio, ``-s 999`` for a deterministic
  reproducible seed, ``--no-flash`` and ``--context 512`` to keep the test
  fast and portable.

  Args:
    cli: path to the installed console script
    models_root: temporary directory containing the model sub-folder

  """
  r: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
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
      '--free',
      '-i',
      str(_IMG_100),
      '-i',
      str(_IMG_107),
      '"describe these images"',
    ],
    text=True,
    capture_output=True,
    check=False,
  )
  # Accept exit 0 (clean) or _MACOS_METAL_CLEANUP_EXIT/-6 (SIGABRT from Metal
  # GPU cleanup on macOS after inference completes successfully).
  assert r.returncode in {0, _MACOS_METAL_CLEANUP_EXIT}, (
    f'Command failed (exit={r.returncode}):\n'
    f'--- stdout ---\n{r.stdout}\n'
    f'--- stderr ---\n{r.stderr}\n'
  )
  # The tiny random model produces garbage, but we verify the full CLI
  # pipeline works end-to-end: model download ➜ load ➜ inference ➜ print.
  assert r.stdout, 'Expected non-empty stdout from model query'
  assert 'portrait' in r.stdout.lower()
  assert '\x1b[' not in r.stdout  # no ANSI escape sequences
  assert '\x1b[' not in r.stderr
