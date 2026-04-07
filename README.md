<!-- SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# TransAI

AI library and helpers (Python/Poetry/Typer - LM Studio & llama.cpp).

- **Primary use case:** Python API/interface with local AI models
- **Works with:** local AI models via [LM Studio](https://lmstudio.ai/) or [llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Status:** stable
- **License:** Apache-2.0

Since version 1.0.0 it is a PyPI package: <https://pypi.org/project/transai/>

## Table of contents

- [TransAI](#transai)
  - [Table of contents](#table-of-contents)
  - [License](#license)
    - [Third-party notices](#third-party-notices)
  - [Installation](#installation)
    - [Supported platforms](#supported-platforms)
    - [Known dependencies (Prerequisites)](#known-dependencies-prerequisites)
  - [What TransAI is](#what-transai-is)
    - [What TransAI is not](#what-transai-is-not)
    - [Key concepts and terminology](#key-concepts-and-terminology)
    - [Known limitations](#known-limitations)
  - [Library API usage](#library-api-usage)
    - [Loading a model](#loading-a-model)
    - [Querying a model (text)](#querying-a-model-text)
    - [Querying a model (structured JSON)](#querying-a-model-structured-json)
    - [Vision models (images)](#vision-models-images)
    - [Image utilities](#image-utilities)
  - [AI Guide](#ai-guide)
    - [Vision Models](#vision-models)
    - [Blind Models](#blind-models)
  - [CLI Interface](#cli-interface)
    - [Quick start](#quick-start)
    - [Global flags](#global-flags)
    - [CLI Commands Documentation](#cli-commands-documentation)
    - [Color and formatting](#color-and-formatting)
  - [Project Design](#project-design)
    - [Architecture overview](#architecture-overview)
    - [Modules](#modules)
  - [Development Instructions](#development-instructions)
    - [File structure](#file-structure)
    - [Development Setup](#development-setup)
      - [Install Python](#install-python)
      - [Install Poetry (recommended: `pipx`)](#install-poetry-recommended-pipx)
      - [Make sure `.venv` is local](#make-sure-venv-is-local)
      - [Get the repository](#get-the-repository)
      - [Create environment and install dependencies](#create-environment-and-install-dependencies)
      - [Optional: VSCode setup](#optional-vscode-setup)
    - [Testing](#testing)
      - [Unit tests / Coverage](#unit-tests--coverage)
      - [Instrumenting your code](#instrumenting-your-code)
      - [Integration / e2e tests](#integration--e2e-tests)
    - [Linting / formatting / static analysis](#linting--formatting--static-analysis)
      - [Type checking](#type-checking)
    - [Versioning and releases](#versioning-and-releases)
      - [Versioning scheme](#versioning-scheme)
      - [Updating versions](#updating-versions)
        - [Bump project version (patch/minor/major)](#bump-project-version-patchminormajor)
        - [Update dependency versions](#update-dependency-versions)
        - [Exporting the `requirements.txt` file](#exporting-the-requirementstxt-file)
        - [CI and docs](#ci-and-docs)
        - [Git tag and commit](#git-tag-and-commit)
        - [Publish to PyPI](#publish-to-pypi)
  - [Security](#security)

## License

Copyright 2025 Daniel Balparda <balparda@github.com>

Licensed under the **Apache License, Version 2.0** (the "License"); you may not use this file except in compliance with the License. You may obtain a [copy of the License here](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### Third-party notices

This project depends on third-party software. Key runtime dependencies:

- [transcrypto](https://github.com/balparda/transcrypto) (Apache-2.0) — CLI modules, logging, utilities
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (MIT) — llama.cpp Python bindings
- [lmstudio](https://pypi.org/project/lmstudio/) — LM Studio client library
- [Pillow](https://github.com/python-pillow/Pillow) (MIT-CMU) — image processing
- [pydantic](https://github.com/pydantic/pydantic) (MIT) — data validation and JSON schema

See `pyproject.toml` for the full dependency list.

## Installation

To use in your project:

```sh
pip3 install transai
```

and then import the library:

```python
from transai.core import ai, lms, llama
from transai.utils import images
```

For the CLI tool, after installation just run:

```sh
transai --help
```

### Supported platforms

- OS: Linux, macOS, Windows (wherever `llama-cpp-python` and `lmstudio` are supported)
- Architectures: x86_64, arm64
- Python: 3.12+

### Known dependencies (Prerequisites)

- **[python 3.12+](https://python.org/)** — [documentation](https://docs.python.org/3.12/)
- **[transcrypto 2.5+](https://pypi.org/project/transcrypto/)** — CLI modules, logging, humanization, config management, etc. — [documentation](https://github.com/balparda/transcrypto)
- **[Pillow 12.2+](https://pypi.org/project/pillow/)** — image processing and format conversion
- **[pydantic 2.12+](https://pypi.org/project/pydantic/)** — data validation and JSON schema generation
- **[llama-cpp-python 0.3.20+](https://pypi.org/project/llama-cpp-python/)** — llama.cpp Python bindings for local GGUF model inference
- **[lmstudio 1.5+](https://pypi.org/project/lmstudio/)** — LM Studio client library for the LM Studio API
- **[rich](https://pypi.org/project/rich/)** — terminal output formatting (via transcrypto)
- **[typer](https://pypi.org/project/typer/)** — CLI framework (via transcrypto)

## What TransAI is

TransAI is a Python library and CLI tool that provides a unified interface for running local AI models through two backends:

- **LM Studio** (`LMStudioWorker`): connects to a running LM Studio server on localhost via the `lmstudio` client library. This is the recommended and default backend.
- **llama.cpp** (`LlamaWorker`): loads GGUF model files directly into memory using `llama-cpp-python`. Useful when you want full control without running an LM Studio server.

Both backends share the same abstract interface (`AIWorker`), so you can swap backends without changing your application code. Models can be queried with plain text prompts or with structured output (Pydantic models), and vision models can process images.

### What TransAI is not

- Not a cloud AI service — it only works with local models
- Not a model downloader — you must have models available locally (via LM Studio or as GGUF files)
- Not a training framework — inference only
- Not a high-level agent framework — it provides the low-level model interface layer

### Key concepts and terminology

- **AIWorker**: abstract base class defining the interface for loading and querying AI models
- **LMStudioWorker**: concrete worker that connects to a local LM Studio server
- **LlamaWorker**: concrete worker that loads GGUF files directly via llama.cpp
- **AIModelConfig**: TypedDict with all model loading parameters (context, temperature, GPU, seed, etc.)
- **Model ID**: a string identifying the model, typically in the format `model-name@quantization` (e.g., `qwen3-8b@Q8_0`); should match what you would use with `lms get <model_id>` or `https://huggingface.co/<model_id>`
- **GGUF**: the quantized model file format used by llama.cpp
- **CLIP projector**: a companion model file enabling vision capabilities in multi-modal models
- **Speculative decoding**: a technique for faster inference by generating multiple tokens in parallel

### Known limitations

- LM Studio backend requires a running LM Studio server on localhost (127.0.0.1)
- llama.cpp backend requires GGUF model files on disk
- Vision support in llama.cpp depends on CLIP projector file availability and supported architectures (Qwen2-VL, MiniCPM, Llama3-Vision, Moondream, NanoLLava, Obsidian, Llava)
- No telemetry, no network calls beyond localhost (LM Studio server)

## Library API usage

### Loading a model

`transai.core.ai` exposes a convenience constructor `MakeAIModelConfig(**overrides)` which
returns a fully-populated `AIModelConfig` TypedDict with sensible defaults.

```python
from transai.core import ai, lms, llama

# --- Using LM Studio ---
with lms.LMStudioWorker() as worker:
  config, metadata = worker.LoadModel(ai.MakeAIModelConfig(
    model_id='qwen3-vl-32b-instruct@Q8_0',
    vision=True,
    temperature=0.5,  # only override the ones you care about!
    # all other fields will have sensible defaults; currently also supported are:
    # seed, context, gpu_ratio, gpu_layers, use_mmap, fp16, flash, spec_tokens, kv_cache
  ))
  # ... use worker.ModelCall() ...

# --- Using llama.cpp ---
import pathlib
with llama.LlamaWorker(pathlib.Path('~/.lmstudio/models/')) as worker:
  config, metadata = worker.LoadModel(ai.AIModelConfig(
    model_id='qwen3-8b@Q8_0',
    # ... same config field possibilities ...
  ))
  # ... use worker.ModelCall() ...
```

### Querying a model (text)

```python
response: str = worker.ModelCall(
  model_id='qwen3-8b@Q8_0',
  system_prompt='You are a helpful assistant.',
  user_prompt='What is the capital of France?',
  output_format=str,
)
print(response)  # "The capital of France is Paris."
```

### Querying a model (structured JSON)

To get a structured object back from the model, just create a `pydantic.BaseModel` class as shown below. Make sure to add pydocs and `pydantic.Field` description to the fields, as all the information (name, type, descriptions) are sent to the model.

```python
import pydantic

class CityInfo(pydantic.BaseModel):
  """City information"""

  city: str = pydantic.Field(description='city name')
  country: str = pydantic.Field(description='country name')
  population: int = pydantic.Field(description='city population')
  districts: list[str] =  pydantic.Field(description='list of city district names')

result: CityInfo = worker.ModelCall(
  model_id='qwen3-8b@Q8_0',
  system_prompt='Extract a city information, its country, population, and list of districts.',
  user_prompt='Tell me about Paris, France.',
  output_format=CityInfo,
)
print(result.city)        # "Paris"
print(result.population)  # 2161000
```

### Vision models (images)

```python
import pathlib

response: str = worker.ModelCall(
  model_id='qwen3-vl-32b-instruct@Q8_0',
  system_prompt='Describe what you see.',
  user_prompt='What is in this image?',
  output_format=str,
  images=[pathlib.Path('photo.jpg')],  # or raw bytes, or file path string
)
```

Images are automatically resized to fit within 1024px (longest edge) before being sent to the model.

### Image utilities

The `transai.utils.images` module provides helpers for image preprocessing:

```python
from transai.utils import images

# Resize an image for vision models (max 1024px, returns PNG bytes)
png_bytes: bytes = images.ResizeImageForVision(raw_image_bytes)

# Extract frames from an animated image (GIF, APNG, etc.)
for frame_png in images.AnimationFrames(animated_gif_bytes):
  # each frame is PNG bytes, resized to max 336px
  pass
```

## AI Guide

Models suggestions as of April/2026. Just an opinion, not to be taken seriously. Do your own tests.

### Vision Models

These models can process images.

| Model Flag Value | Size | Type | Tool? | Reason? | Comment |
| --- | --- | --- | --- | --- | --- |
| [`qwen3-vl-32b-instruct@Q8_0`](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF) | 36GB | `llm/qwen3vl/GGUF` | Y | | Very good, slow. |
| [`qwen3-vl-32b-instruct@F16`](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF) | 67GB | `llm/qwen3vl/GGUF` | Y | | `--fp16` - Very good, slow.  Q8_0 version is faster-ish and still very good. |
| [`qwen3.5-35b-a3b@Q8_0`](https://lmstudio.ai/models/qwen/qwen3.5-35b-a3b) [*](https://huggingface.co/lmstudio-community/Qwen3.5-35B-A3B-GGUF) | 38GB | `llm/qwen35moe/GGUF` | Y | Y | Decent, slow. |
| [`zai-org/glm-4.6v-flash@8bit`](https://lmstudio.ai/models/zai-org/glm-4.6v-flash) [*](https://huggingface.co/lmstudio-community/GLM-4.6V-Flash-MLX-8bit) | 12GB | `llm/glm4v/MLX` | Y | Y | Decent, slow. |

### Blind Models

These models cannot process images (blind).

| Model Flag Value | Size | Type | Tool? | Reason? | Comment |
| --- | --- | --- | --- | --- | --- |
| [`qwen3-8b@Q8_0`](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | 8.7GB | `llm/qwen3/GGUF` | Y | | Good, medium-speed. |
| [`gpt-oss-20b@MXFP4`](https://lmstudio.ai/models/openai/gpt-oss-20b) [*](https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8) | 12GB | `llm/gpt_oss/MLX` | Y | Y | Poor, slow. |
| [`zai-org/glm-4.7-flash@8bit`](https://lmstudio.ai/models/zai-org/glm-4.7-flash) [*](https://huggingface.co/lmstudio-community/GLM-4.7-Flash-MLX-8bit) | 32GB | `llm/glm4v/MLX` | Y | Y | Good, inconsistent. |

## CLI Interface

### Quick start

Query a local AI model via LM Studio (server must be running):

```sh
transai query "What is the capital of France?"
```

Query using the llama.cpp backend (direct GGUF loading, no server needed):

```sh
transai --no-lms --root ~/.lmstudio/models/ query "Give me an onion soup recipe."
```

### Global flags

| Flag | Description | Default |
| --- | --- | --- |
| `--help` | Show help | off |
| `--version` | Show version and exit | off |
| `-v`, `-vv`, `-vvv`, `--verbose` | Verbosity (nothing=*ERROR*, `-v`=*WARNING*, `-vv`=*INFO*, `-vvv`=*DEBUG*) | *ERROR* |
| `--color`/`--no-color` | Force enable/disable colored output (respects `NO_COLOR` env var if not provided) | `--color` |
| `-r`, `--root` | Local models root directory (only needed for `--no-lms`) | LM Studio default if it exists |
| `--lms`/`--no-lms` | Use LM Studio backend vs llama.cpp backend | `--lms` |
| `-m`, `--model` | Model to load (e.g., `qwen3-8b@Q8_0`) | `qwen3-8b@Q8_0` |
| `-t`, `--tokens` | Speculative decoding tokens (2-200) | disabled |
| `-s`, `--seed` | Random seed for reproducibility | random |
| `--context` | Max context tokens (16-16777216) | 32768 |
| `-x`, `--temperature` | Sampling temperature (0.0-2.0) | 0.15 |
| `-g`, `--gpu` | GPU ratio (0.1-1.0) | 0.80 |
| `--gpu-layers` | GPU layers to offload (-1 = as many as possible) | -1 |
| `--fp16`/`--no-fp16` | FP16 precision mode | `--no-fp16` |
| `--mmap`/`--no-mmap` | Memory-mapped file loading | `--mmap` |
| `--flash`/`--no-flash` | Flash attention | `--flash` |
| `--kv-cache` | KV-cache precision type (GGML type, 4-128) | model default |

### CLI Commands Documentation

This software auto-generates docs for CLI apps:

- [**`transai`** documentation](transai.md)

### Color and formatting

Rich provides color output in logging and CLI output. The app:

- Respects `NO_COLOR` environment variable
- Has `--no-color` / `--color` flag: if given, overrides the `NO_COLOR` environment variable
- If there is no environment variable and no flag is given, defaults to having color

To control color see [Rich's markup conventions](https://rich.readthedocs.io/en/latest/markup.html).

## Project Design

### Architecture overview

TransAI uses an abstract base class pattern for backend abstraction:

```txt
CLI (transai.py + cli/query.py)
  │
  ├─ LMStudioWorker (core/lms.py)  ──▶  LM Studio server (localhost)
  │
  └─ LlamaWorker (core/llama.py)   ──▶  GGUF files on disk
  │
  └─ Both implement AIWorker (core/ai.py)
       │
       └─ Image utilities (utils/images.py)
```

- `AIWorker` defines `LoadModel()` and `ModelCall()` as the public interface
- `LMStudioWorker` and `LlamaWorker` implement `_Load()` and `_Call()` internally
- The CLI layer (`transai.py`, `cli/query.py`) orchestrates configuration and delegates to workers
- Image preprocessing is handled by `utils/images.py`

### Modules

| Module | Responsibility |
| --- | --- |
| `transai.py` | CLI app definition, global options, `TransAIConfig` dataclass |
| `cli/query.py` | `query` command implementation |
| `core/ai.py` | `AIWorker` abstract base class, `AIModelConfig`, shared constants and types |
| `core/lms.py` | `LMStudioWorker` — LM Studio backend implementation |
| `core/llama.py` | `LlamaWorker` — llama.cpp backend implementation (GGUF loading, CLIP detection, vision handlers) |
| `utils/images.py` | Image resizing for vision models, animation frame extraction |

## Development Instructions

### File structure

```txt
.
├── CHANGELOG.md                  ⟸ latest changes/releases
├── LICENSE
├── Makefile
├── transai.md                    ⟸ auto-generated CLI doc (by `make docs` or `make ci`)
├── poetry.lock                   ⟸ maintained by Poetry, do not manually edit
├── pyproject.toml                ⟸ most important configurations live here
├── README.md                     ⟸ this documentation
├── SECURITY.md                   ⟸ security policy
├── requirements.txt
├── .pre-commit-config.yaml       ⟸ pre-submit configs
├── .github/
│   ├── copilot-instructions.md   ⟸ GitHub Copilot project-specific instructions
│   ├── dependabot.yaml           ⟸ Github dependency update pipeline
│   └── workflows/
│       ├── ci.yaml               ⟸ Github CI pipeline
│       └── codeql.yaml           ⟸ Github security scans and code quality pipeline
├── .vscode/
│   └── settings.json             ⟸ VSCode configs
├── scripts/
│   └── make_test_images.py       ⟸ helper script for generating test images
├── src/
│   └── transai/
│       ├── __init__.py           ⟸ version and package metadata
│       ├── __main__.py           ⟸ `python -m transai` entry point
│       ├── transai.py            ⟸ main CLI app entry point (Run(), Main())
│       ├── py.typed              ⟸ PEP 561 marker for type stubs
│       ├── cli/
│       │   └── query.py          ⟸ `transai query` command implementation
│       ├── core/
│       │   ├── ai.py             ⟸ AIWorker abstract base class, AIModelConfig, shared types
│       │   ├── llama.py          ⟸ LlamaWorker (llama.cpp backend)
│       │   └── lms.py            ⟸ LMStudioWorker (LM Studio backend)
│       └── utils/
│           └── images.py         ⟸ image preprocessing for vision models
├── tests/                        ⟸ unit tests
│   ├── transai_test.py
│   ├── cli/
│   │   └── query_test.py
│   ├── core/
│   │   ├── ai_test.py
│   │   ├── llama_test.py
│   │   └── lms_test.py
│   └── utils/
│       └── images_test.py
└── tests_integration/
    └── test_installed_cli.py     ⟸ integration tests (wheel build + install)
```

### Development Setup

#### Install Python

On **Linux**:

```sh
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git python3 python3-dev python3-venv build-essential software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.12
```

On **Mac**:

```sh
brew update
brew upgrade
brew cleanup -s

brew install git python@3.12
```

#### Install Poetry (recommended: `pipx`)

[Poetry reference.](https://python-poetry.org/docs/cli/)

Install `pipx` (if you don't have it):

```sh
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

If you previously had **Poetry** installed, but ***not*** through `pipx` make sure to remove it first: `brew uninstall poetry` (mac) / `sudo apt-get remove python3-poetry` (linux). You should install Poetry with `pipx` and configure poetry to create `.venv/` locally. This keeps Poetry isolated from project virtual environments and python for the environments is isolated from python for Poetry. Do:

```sh
pipx install poetry
poetry --version
```

If you will use [PyPI](https://pypi.org/) to publish:

```sh
poetry config pypi-token.pypi <TOKEN>  # add your personal PyPI project token, if any
```

#### Make sure `.venv` is local

This project expects a project-local virtual environment at `./.venv` (VSCode settings assume it).

```sh
poetry config virtualenvs.in-project true
```

#### Get the repository

```sh
git clone https://github.com/balparda/transai.git transai
cd transai
```

#### Create environment and install dependencies

From the repository root:

```sh
poetry env use python3.12  # creates the .venv with the correct Python version
poetry sync                # sync env to project's poetry.lock file
poetry env info            # no-op: just to check that environment looks good
poetry check               # no-op: make sure all pyproject.toml fields are being used correctly

poetry run transai --help    # simple test if everything loaded OK
make ci                    # should pass OK on clean repo
```

To activate and use the environment do:

```sh
poetry env activate        # (optional) will print activation command for environment, but you can just use:
source .venv/bin/activate  # because .venv SHOULD BE LOCAL
...
pytest -vvv  # for example, or other commands you want to execute in-environment
...
deactivate  # to close environment
```

#### Optional: VSCode setup

This repo ships a `.vscode/settings.json` configured to:

- use `./.venv/bin/python`
- run `pytest`
- use **Ruff** as formatter
- disable deprecated pylint/flake8 integrations
- configure Google-style docstrings via **autoDocstring**
- use **Code Spell Checker**

Recommended VSCode extensions:

- Python (`ms-python.python`)
- Python Environments (`ms-python.vscode-python-envs`)
- Python Debugger (`ms-python.debugpy`)
- Pylance (`ms-python.vscode-pylance`)
- Mypy Type Checker (`ms-python.mypy-type-checker`)
- Ruff (`charliermarsh.ruff`)
- autoDocstring – Python Docstring Generator (`njpwerner.autodocstring`)
- Code Spell Checker (`streetsidesoftware.code-spell-checker`)
- markdownlint (`davidanson.vscode-markdownlint`)
- Markdown All in One (`yzhang.markdown-all-in-one`) - helps maintain this `README.md` table of contents
- Markdown Preview Enhanced (`shd101wyy.markdown-preview-enhanced`, optional)
- GitHub Copilot (`github.copilot`) - AI assistant; reads `.github/copilot-instructions.md` for project-specific coding conventions (indentation, naming, workflow)

### Testing

#### Unit tests / Coverage

```sh
make test               # plain test run, no integration tests
make integration        # run the integration tests
poetry run pytest -vvv  # verbose test run, includes integration tests

make cov  # coverage run, equivalent to: poetry run pytest --cov=src --cov-report=term-missing
```

A test can be marked with a "tag" by just adding a decorator:

```python
@pytest.mark.slow
def test_foo_method() -> None:
  """Test."""
  ...
```

These tags are defined in `pyproject.toml`, in section `[tool.pytest.ini_options.markers]`:

| Tag | Meaning |
| --- | --- |
| `slow` | test is slow (> 1s) |
| `flaky` | AVOID! — test is known to be flaky |
| `stochastic` | test is capable of failing (even if very unlikely) |
| `integration` | integration test (wheel build + install) |

You can use them to filter tests:

```sh
poetry run pytest -vvv -m slow  # run only the slow tests
```

You can find the slowest tests by running:

```sh
poetry run pytest -vvv -q --durations=20
```

You can search for flaky tests by running `make flakes`, which runs all tests 100 times.

#### Instrumenting your code

You can instrument your code to find bottlenecks:

```sh
$ source .venv/bin/activate
$ which transai
/path/to/.venv/bin/transai  # <== place this in the command below:
$ pyinstrument -r html -o output1.html -- /path/to/.venv/bin/transai <your-cli-command> <your-cli-flags>
$ deactivate
```

This will save a file `output1.html` to the project directory with the timings for all method calls. Make sure to **cleanup** these html files later.

#### Integration / e2e tests

Integration tests validate packaging and the installed console script by:

- building a wheel from the repository
- installing that wheel into a fresh temporary virtualenv
- running the installed console script(s) to verify behavior (e.g., `--version` and basic commands)

The canonical integration test is [tests_integration/test_installed_cli.py](tests_integration/test_installed_cli.py). Tests in this suite are marked with `pytest.mark.integration`.

Run the integration tests with:

```sh
make integration  # or: poetry run pytest -m integration -q
```

### Linting / formatting / static analysis

```sh
make lint  # equivalent to: poetry run ruff check .
make fmt   # equivalent to: poetry run ruff format .
```

To check formatting without rewriting:

```sh
poetry run ruff format --check .
```

#### Type checking

```sh
make type  # equivalent to: poetry run mypy src tests tests_integration
```

(Pyright is primarily for editor-time; MyPy is what CI enforces.)

### Versioning and releases

#### Versioning scheme

This project follows a pragmatic versioning approach:

- **Patch**: bug fixes / docs / small improvements.
- **Minor**: new features or non-breaking changes.
- **Major**: breaking API changes.

See: [CHANGELOG.md](CHANGELOG.md)

#### Updating versions

##### Bump project version (patch/minor/major)

Poetry can bump versions:

```sh
# bump the version!
poetry version minor  # updates 1.0.0 to 1.1.0, for example
# or:
poetry version patch  # updates 1.0.0 to 1.0.1
# or:
poetry version <version-number>
# (also updates `pyproject.toml` and `poetry.lock`)
```

This updates `[project].version` in `pyproject.toml`. **Remember to also update `src/transai/__init__.py` to match (this repo gets/prints `__version__` from there)!**

##### Update dependency versions

The project has a [**dependabot**](https://docs.github.com/en/code-security/tutorials/secure-your-dependencies/dependabot-quickstart-guide) config file in `.github/dependabot.yaml` that weekly (defaulting to Tuesdays) scans both Github actions and the project dependencies and creates PRs to update them.

To update `poetry.lock` file to more current versions do `poetry update`, it will ignore the current lock, update, and rewrite the `poetry.lock` file. If you have cache problems `poetry cache clear PyPI --all` will clean it.

To add a new dependency you should do:

```sh
poetry add "pkg>=1.2.3"  # regenerates lock, updates env (adds dep to prod code)
poetry add -G dev "pkg>=1.2.3"  # adds dep to dev code ("group" dev)
# also remember: "pkg@^1.2.3" = latest 1.* ; "pkg@~1.2.3" = latest 1.2.* ; "pkg@1.2.3" exact
```

Keep tool versions aligned. Remember to check your diffs before submitting (especially `poetry.lock`) to avoid surprises!

##### Exporting the `requirements.txt` file

This project does not generate `requirements.txt` automatically (Poetry uses `poetry.lock`). If you need a `requirements.txt` for Docker/legacy tooling, use Poetry's export plugin (`poetry-plugin-export`) by simply running:

```sh
make req  # or: poetry export --format requirements.txt --without-hashes --output requirements.txt
```

##### CI and docs

Make sure to run `make docs` or even better `make ci`. Both will update the CLI markdown docs and `requirements.txt` automatically.

##### Git tag and commit

Publish to GIT, including a TAG:

```sh
git commit -a -m "release version 1.0.0"
git tag 1.0.0
git push
git push --tags
```

##### Publish to PyPI

If you already have your PyPI token registered with Poetry (see [Install Poetry](#install-poetry-recommended-pipx)) then just:

```sh
poetry build
poetry publish
```

Remember to update [CHANGELOG.md](CHANGELOG.md).

## Security

Please refer to the security policy in [SECURITY.md](SECURITY.md) for supported versions and how to report vulnerabilities.

The project has a [**codeql**](https://codeql.github.com/docs/) config file in `.github/workflows/codeql.yaml` that weekly (defaulting to Fridays) scans the project for code quality and security issues. It will also run on all commits. Github security issues will be opened in the project if anything is found.
