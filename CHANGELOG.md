<!-- SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Changelog

All notable changes to this project will be documented in this file.

- [Changelog](#changelog)
  - [V.V.V - YYYY-MM-DD - Placeholder](#vvv---yyyy-mm-dd---placeholder)
  - [1.1.0 - 2026-??-?? - Placeholder](#110---2026-----placeholder)
  - [1.0.0 - 2026-04-05](#100---2026-04-05)

This project follows a pragmatic versioning approach:

- **Patch**: bug fixes / docs / small improvements.
- **Minor**: new template features or non-breaking developer workflow changes.
- **Major**: breaking template changes (e.g., required file/command renames).

## V.V.V - YYYY-MM-DD - Placeholder

- Added
  - Tool calling support in both `LMStudioWorker` (via `_CallLMSAct`) and `LlamaWorker` (via `_CallLlamaAct`): models that support function calling can now invoke Python callables passed via the `tools` parameter of `ModelCall()`
  - `--tools` CLI option added to the `query` command: pass one or more fully-qualified Python callable names (e.g., `--tools math.gcd --tools os.getcwd`)
  - `LoadModel()` and `ModelCall()` now enforce a `--timeout` option

- Changed
  - Placeholder for future changes.

- Fixed
  - Placeholder for future changes.

## 1.1.0 - 2026-??-?? - Placeholder

- Added
  - integration test that downloads big model, to run locally (1.0.1)

- Changed
  - load() and call() model will catch Exception and raise Error

- Fixed
  - N/A

## 1.0.0 - 2026-04-05

Initial public release. Published to [pypi.org](https://pypi.org/project/transai/) on 2026-04-05. Features:

- Two AI backends with a unified abstract interface (`AIWorker`):
  - **LM Studio** backend (`LMStudioWorker`): connects to a local LM Studio server via the `lmstudio` client library. Auto-discovers the local API host and validates loopback-only connections.
  - **llama.cpp** backend (`LlamaWorker`): loads GGUF model files directly via `llama-cpp-python`. Auto-discovers model directories, GGUF files, and CLIP projectors.
- Model loading with extensive configuration options:
  - GPU offload ratio (`--gpu`) and layer count (`--gpu-layers`)
  - Context length (`--context`, default 32k tokens)
  - Sampling temperature (`--temperature`)
  - Reproducible results via seed (`--seed`)
  - Speculative decoding (`--tokens`)
  - FP16 precision, flash attention, memory-mapped loading, and KV-cache type
- Model calling with two output formats:
  - Plain text (`str`) responses
  - Structured JSON output via Pydantic model classes (uses JSON Schema for llama.cpp, `response_format` for LM Studio)
- Vision model support:
  - Automatic CLIP projector detection for llama.cpp (Qwen2-VL, MiniCPM, Llama3-Vision, Moondream, NanoLLava, Obsidian, Llava)
  - Image preprocessing: automatic resize to 1024px max dimension
  - Animated image support: frame extraction with decimation to ~10 frames
  - Multiple image input types: `bytes`, `pathlib.Path`, or `str` (file path)
- Capability detection: automatic identification of vision, tooling, and reasoning capabilities from model metadata
- `transai query` CLI command for quick interactive queries
- `transai markdown` CLI command for auto-generating CLI documentation
- Global CLI flags: `--version`, `--verbose`, `--color`/`--no-color`, `--lms`/`--no-lms`, `--model`, `--root`
- Full type annotation coverage (MyPy strict + Pyright strict + typeguard)
- Unit tests and integration tests (wheel build + install + smoke tests)
- CI pipeline with linting, type checking, coverage, and integration tests
