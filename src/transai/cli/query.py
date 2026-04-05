# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""CLI: AI Query."""

from __future__ import annotations

import logging

import click
import typer
from transcrypto.cli import clibase
from transcrypto.utils import timer

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
def IsPrimeCLI(  # documentation is help/epilog/args # noqa: D103
  *,
  ctx: click.Context,
  model_input: str = typer.Argument(..., help='Model input string'),
) -> None:
  config: transai.TransAIConfig = ctx.obj
  if not config.lms and not config.models_root:
    raise ai.Error('Non-LM Studio client library requires `models_root` to be set')
  with timer.Timer('Model LOAD'):
    worker: ai.AIWorker = (
      lms.LMStudioWorker(free_resources=False)
      if config.lms
      else llama.LlamaWorker(config.models_root, verbose=config.verbose < logging.INFO)  # type: ignore[arg-type]
    )
    model_config, _ = worker.LoadModel(
      ai.MakeAIModelConfig(
        model_id=config.model,
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
  with timer.Timer('Model QUERY'):
    response: str = worker.ModelCall(model_config['model_id'], '', model_input, str)
  config.console.print(response)
