# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""func_timeout minimal type stub."""

from typing import Callable

__version__ = ...
__version_tuple__ = ...
__all__ = ('func_timeout',)

def func_timeout[T: object](timeout: float, func: Callable[..., T], args: list[object] | None = ..., kwargs: dict[str, object] | None = ...) -> T: ...
