# SPDX-FileCopyrightText: 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0

.PHONY: install init fmt lint type test integration cov flakes precommit docs req build ci

install:
	poetry install

init:
	@echo "Initializing Poetry environment with in-project virtualenv and Python 3.12"
	poetry config virtualenvs.in-project true
	poetry env use python3.12
	poetry sync

fmt:
	poetry run ruff format .

lint:
	poetry run ruff check .

type:
	poetry run mypy src tests tests_integration

test:
	poetry run pytest -vvvv -q tests

integration:
	poetry run pytest -vvvv -q tests_integration

cov:
	poetry run pytest --typeguard-packages=transai --cov=src --cov-report=term-missing -q tests

flakes:
	poetry run pytest --flake-finder --flake-runs=100 -q tests

precommit:
	poetry run pre-commit run --all-files

docs:
	@echo "Generating transai.md"
	poetry run transai markdown > transai.md

req:
	@echo "Generating requirements.txt from Poetry dependencies"
	poetry export --format requirements.txt --without-hashes --output requirements.txt

build:
	@echo "Building source and wheel distributions with Poetry"
	poetry build --clean -vv

ci: build cov integration precommit docs req
	@echo "CI checks passed! Generated docs & requirements.txt."
