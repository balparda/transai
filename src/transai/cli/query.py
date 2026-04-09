# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""CLI: AI Query."""

from __future__ import annotations

import logging
import pathlib

import click
import typer
from transcrypto.cli import clibase

from transai import transai
from transai.core import ai, llama, lms


@transai.app.command(
  'query',
  help='Query the model.',
  epilog=(
    'Example:\n\n\n\n'
    'poetry run transai query "What is the capital of France?"\n\n'
    'poetry run transai --no-lms query "Give me an onion soup recipe."'
  ),
)
@clibase.CLIErrorGuard
def Query(  # documentation is help/epilog/args # noqa: D103
  *,
  ctx: click.Context,
  model_input: str = typer.Argument(..., help='Query input string; "user prompt"'),
  system_prompt: str = typer.Option(
    '',
    '-y',
    '--system',
    help=('Prefix prompt; prepend to query; "system prompt"; default: no system prompt'),
  ),
  images: list[pathlib.Path] | None = typer.Option(  # noqa: B008
    None,
    '-i',
    '--images',
    exists=True,
    file_okay=True,
    readable=True,
    help=('A list of image paths to use as input for the model query; default: None, no images'),
  ),
  tools: list[str] | None = typer.Option(  # noqa: B008
    None,
    '-z',
    '--tools',
    help=('A list of python methods to use as tools for the model query; default: None, no tools'),
  ),
  free_resources: bool = typer.Option(
    False,
    '--free/--no-free',
    help=('Unload previous models before loading new ones (LM Studio)? default: False (keep)'),
  ),
  metal: bool = typer.Option(
    False,
    '--metal/--no-metal',
    help=('Print Metal/llama.cpp verbose internals? default: False (do not print)'),
  ),
) -> None:
  config: transai.TransAIConfig = ctx.obj
  if not config.lms and not config.models_root:
    raise ai.Error('Non-LM Studio client library requires `models_root` to be set')
  if not free_resources and config.seed is not None:
    logging.warning(f'Seed {config.seed} + `--no-free`, but to apply seed we will `--free`')
  # use the worker as a context manager so Close() is called explicitly before process exit,
  # which ensures llama_cpp.Llama (and its Metal/GPU resources) are freed while the runtime
  # is still in a valid state; without this, Metal cleanup during __cxa_finalize_ranges
  # (C++ static destructors at exit) can trigger a ggml_metal_device_free assertion failure
  with (
    lms.LMStudioWorker(
      timeout=config.timeout, free_resources=free_resources or config.seed is not None
    )
    if config.lms
    else llama.LlamaWorker(config.models_root, timeout=config.timeout, verbose=metal)  # type: ignore[arg-type]
  ) as ai_worker:
    model_config, _ = ai_worker.LoadModel(
      ai.MakeAIModelConfig(
        model_id=config.model,
        vision=bool(images),  # if there are images, we need vision support!
        seed=config.seed,
        context=config.context,
        temperature=config.temperature,
        gpu_ratio=config.gpu,
        gpu_layers=config.gpu_layers,
        use_mmap=config.use_mmap,
        fp16=config.fp16,
        flash=config.flash,
        spec_tokens=config.spec_tokens,
        kv_cache=config.kv_cache,
      )
    )
    response: str = ai_worker.ModelCall(
      model_config['model_id'],
      system_prompt.strip(),
      model_input.strip(),
      str,
      images=list(images) if images else None,
      tools=tools,  # type: ignore[arg-type]
    )
    config.console.print(response)
    config.console.print()
