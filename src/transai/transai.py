# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Simple CLI into TransAI library methods."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import click
import typer
from rich import console as rich_console
from transcrypto.cli import clibase
from transcrypto.utils import config as app_config
from transcrypto.utils import human
from transcrypto.utils import logging as cli_logging

from transai.core import ai

from . import __version__


@dataclass(kw_only=True, slots=True, frozen=True)
class TransAIConfig(clibase.CLIConfig):
  """TransAI global context, storing the configuration."""

  lms: bool
  model: str
  seed: int | None
  context: int
  temperature: float
  gpu: float
  fp16: bool
  spec_tokens: int | None
  models_root: pathlib.Path | None
  gpu_layers: int
  use_mmap: bool
  flash: bool
  kv_cache: int | None
  timeout: float | None


MODELS_ROOT_OPTION: typer.models.OptionInfo = typer.Option(
  ai.DEFAULT_MODELS_ROOT,
  '-r',
  '--root',
  exists=True,
  file_okay=False,
  dir_okay=True,
  readable=True,
  help=(
    'The local machine models root directory path, ex: "~/.lmstudio/models/"; '
    'will expect the models to be in directories under this given root, '
    'usually the hierarchy looks like: <root>/<user>/<model-dir>/*.gguf; '
    'only necessary for non-LMStudio (`--no-lms`) runs; '
    'default: the LMStudio models root if it exists, otherwise no default and you must provide it'
  ),
)
LMS_OPTION: typer.models.OptionInfo = typer.Option(
  True,  # TODO: change to False??
  '--lms/--no-lms',
  help=(
    'Use LMStudio client library for AI instead of the old llama-cpp-python library? '
    'default: True (LMStudio)'
  ),
)
MODEL_OPTION: typer.models.OptionInfo = typer.Option(
  ai.DEFAULT_TEXT_MODEL,
  '-m',
  '--model',
  help=(
    'LLM model to load and use: '
    'the model must be compatible with the llama.cpp/LMStudio client libraries; '
    'will NOT get the model for you, so make sure you either have it available in your LMStudio '
    'or the model files are under the specified models root path (`-r/--root` option); '
    'should be a string you would use with `lms get <THIS>` or `https://huggingface.co/<THIS>`; '
    f'default: {ai.DEFAULT_TEXT_MODEL!r}, a good general-purpose text (non-vision) model'
  ),
)
SPEC_TOKENS_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '-t',
  '--tokens',
  min=2,
  max=200,
  help=(
    'Speculative Decoding: controls how many tokens the model should generate in advance during '
    'auto-tagging; if you do not define this flag then speculative decoding will be disabled; '
    'usually this is a small value, like 4 or 8, and it can improve the speed of processing '
    'by allowing the model to generate tokens in parallel; '
    'default: None (disabled)'
  ),
)
SEED_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '-s',
  '--seed',
  min=2,
  max=ai.AI_MAX_SEED,
  help=(
    'A seed value for the random number generator used to load the models into memory; '
    'providing a seed ensures reproducibility of the results; '
    'default: None (randomized seed)'
  ),
)
CONTEXT_OPTION: typer.models.OptionInfo = typer.Option(
  ai.AI_CONTEXT_LENGTH,
  '--context',
  min=16,
  max=ai.AI_MAX_CONTEXT,
  help=(
    'Maximum number of tokens to use as context for the model; '
    f'default: {ai.AI_CONTEXT_LENGTH} tokens'
  ),
)
TEMPERATURE_OPTION: typer.models.OptionInfo = typer.Option(
  ai.DEFAULT_TEMPERATURE,
  '-x',
  '--temperature',
  min=0.0,
  max=ai.MAX_TEMPERATURE,
  help=(
    'Temperature controls how random token selection is during generation; '
    '[0 or near 0]: most deterministic, focused, repetitive, best for extraction / '
    'structured output / coding / tool use; [0.2-0.5]: still stable, but less rigid; '
    '[0.7-1.0]: more natural and varied; [>1.0]: often more creative, but also '
    'more errors, drift, and nonsense; '
    f'default: {ai.DEFAULT_TEMPERATURE:.3f} (a good value for structured output and tool use)'
  ),
)
GPU_OPTION: typer.models.OptionInfo = typer.Option(
  ai.DEFAULT_GPU_RATIO,
  '-g',
  '--gpu',
  min=0.1,
  max=1.0,
  help=(
    'GPU ratio to use, a value between 0.1 (10%) and 1.0 (100%) that '
    'indicates the percentage of GPU resources to allocate to AI; '
    f'default: {ai.DEFAULT_GPU_RATIO:.2f}'
  ),
)
GPU_LAYERS_OPTION: typer.models.OptionInfo = typer.Option(
  -1,
  '--gpu-layers',
  min=-1,
  max=128,
  help=('Number of layers offloaded to GPU; default: -1 (which means "as many as possible")'),
)
FP16_OPTION: typer.models.OptionInfo = typer.Option(
  False,
  '--fp16/--no-fp16',
  help=(
    'Use FP16 precision for the auto-tagger model? '
    'This can reduce memory usage and potentially increase speed, '
    'but may slightly affect the accuracy of the tagging results '
    'default: False (do not use FP16, use full precision)'
  ),
)
USE_MMAP_OPTION: typer.models.OptionInfo = typer.Option(
  True,
  '--mmap/--no-mmap',
  help=('Use memory-mapped file loading (if supported)? default: True (use mmap)'),
)
FLASH_OPTION: typer.models.OptionInfo = typer.Option(
  True,
  '--flash/--no-flash',
  help=('Enable flash attention (if supported)? default: True (use flash)'),
)
KV_CACHE_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '--kv-cache',
  min=4,
  max=128,
  help=(
    'GGML type for KV-cache keys/values (if supported): '
    'determines the precision level used to store keys/values; '
    'default: None (store according to original precision in model)'
  ),
)
TIMEOUT_OPTION: typer.models.OptionInfo = typer.Option(
  ai.DEFAULT_TIMEOUT,
  '--timeout',
  min=1.0,
  max=24 * 60.0 * 60.0,  # up to 24 hours
  help=(
    'Timeout in seconds for model loading and calls; '
    f'default: {human.HumanizedSeconds(ai.DEFAULT_TIMEOUT)}'
  ),
)


# CLI app setup, this is an important object and can be imported elsewhere and called
app = typer.Typer(
  add_completion=True,
  no_args_is_help=True,
  # keep in sync with Main().help
  help='AI library and helpers (Python/Poetry/Typer - LM Studio & llama.cpp)',
  epilog=(
    'Examples:\n\n\n\n'
    '# --- Query the AI ---\n\n'
    'poetry run transai query "What is the capital of France?"\n\n'
    'poetry run transai --no-lms query "Give me an onion soup recipe."\n\n\n\n'
    '# --- Markdown ---\n\n'
    'poetry run transai markdown > transai.md'
  ),
)


def Run() -> None:
  """Run the CLI."""
  app()


@app.callback(
  invoke_without_command=True,
  help='AI library and helpers (Python/Poetry/Typer - LM Studio & llama.cpp)',
)  # keep message in sync with app.help
@clibase.CLIErrorGuard
def Main(  # documentation is help/epilog/args # noqa: D103
  *,
  ctx: click.Context,  # global context
  version: bool = typer.Option(False, '--version', help='Show version and exit.'),
  verbose: int = typer.Option(
    0,
    '-v',
    '--verbose',
    count=True,
    help='Verbosity (nothing=ERROR, -v=WARNING, -vv=INFO, -vvv=DEBUG).',
    min=0,
    max=3,
  ),
  color: bool | None = typer.Option(
    None,
    '--color/--no-color',
    help=(
      'Force enable/disable colored output (respects NO_COLOR env var if not provided). '
      'Defaults to having colors.'  # state default because None default means docs don't show it
    ),
  ),
  models_root: pathlib.Path | None = MODELS_ROOT_OPTION,  # type: ignore[assignment]
  lms: bool = LMS_OPTION,  # type: ignore[assignment]
  model: str = MODEL_OPTION,  # type: ignore[assignment]
  spec_tokens: int | None = SPEC_TOKENS_OPTION,  # type: ignore[assignment]
  seed: int | None = SEED_OPTION,  # type: ignore[assignment]
  context: int = CONTEXT_OPTION,  # type: ignore[assignment]
  temperature: float = TEMPERATURE_OPTION,  # type: ignore[assignment]
  gpu: float = GPU_OPTION,  # type: ignore[assignment]
  gpu_layers: int = GPU_LAYERS_OPTION,  # type: ignore[assignment]
  fp16: bool = FP16_OPTION,  # type: ignore[assignment]
  use_mmap: bool = USE_MMAP_OPTION,  # type: ignore[assignment]
  flash: bool = FLASH_OPTION,  # type: ignore[assignment]
  kv_cache: int | None = KV_CACHE_OPTION,  # type: ignore[assignment]
  timeout: float | None = TIMEOUT_OPTION,  # type: ignore[assignment]
) -> None:
  if version:
    typer.echo(__version__)
    raise typer.Exit(0)
  console: rich_console.Console
  console, verbose, color = cli_logging.InitLogging(
    verbose,
    color=color,
    include_process=False,  # decide if you want process names in logs
    soft_wrap=False,  # decide if you want soft wrapping of long lines
  )
  # create context with the arguments we received
  ctx.obj = TransAIConfig(
    console=console,
    verbose=verbose,
    color=color,
    appconfig=app_config.InitConfig('transai', 'config.bin'),
    lms=lms,
    model=model,
    seed=seed,
    context=context,
    temperature=temperature,
    gpu=gpu,
    fp16=fp16,
    spec_tokens=spec_tokens,
    models_root=models_root,
    gpu_layers=gpu_layers,
    use_mmap=use_mmap,
    flash=flash,
    kv_cache=kv_cache,
    timeout=timeout,
  )
  # even though this is a convenient place to print(), beware that this runs even when
  # a subcommand is invoked; so prefer logging.debug/info/warning/error instead of print();
  # for example, if you run `markdown` subcommand, this will still print and spoil the output


@app.command(
  'markdown',
  help='Emit Markdown docs for the CLI (see README.md section "Creating a New Version").',
  epilog=('Example:\n\n\n\n$ poetry run transai markdown > transai.md\n\n<<saves CLI doc>>'),
)
@clibase.CLIErrorGuard
def Markdown(*, ctx: click.Context) -> None:  # documentation is help/epilog/args # noqa: D103
  config: TransAIConfig = ctx.obj
  config.console.print(clibase.GenerateTyperHelpMarkdown(app, prog_name='transai'))


# Import CLI modules to register their commands with the app
from transai.cli import query  # pyright: ignore[reportUnusedImport] # noqa: E402, F401
