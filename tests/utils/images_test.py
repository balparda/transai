# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""images.py unittest.

Run with:
  poetry run pytest -vvv tests/utils/images_test.py
"""

from __future__ import annotations

import pathlib

_TEST_IMAGES_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / 'data' / 'images'
