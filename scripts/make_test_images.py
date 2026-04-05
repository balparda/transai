#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Generate images for tests.

Usage
poetry run scripts/make_test_images.py

Notes
-----
- Keep logic thin here; import and call into src/ code.

"""

from __future__ import annotations

import pathlib

# import pdb
from transcrypto.core import hashes

from transai.utils import images

_TEST_IMAGES_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / 'tests' / 'data' / 'images'
_IMG_100: pathlib.Path = _TEST_IMAGES_PATH / '100.jpg'
_IMG_109: pathlib.Path = _TEST_IMAGES_PATH / '109.gif'


def Main() -> int:
  """Generate images.

  Returns:
    int: Exit code

  """
  # pdb.set_trace()
  assert _IMG_100.exists(), f'{_IMG_100} does not exist; please add it to the repo'
  assert _IMG_109.exists(), f'{_IMG_109} does not exist; please add it to the repo'
  # do 100
  b100: bytes = _IMG_100.read_bytes()
  p100 = pathlib.Path(_TEST_IMAGES_PATH / '100-reduced-128.png')
  p100.write_bytes(images.ResizeImageForVision(b100, max_pixels=128))
  print(f'Wrote {p100} : {hashes.Hash256(p100.read_bytes()).hex()}')
  # do 109
  b109: bytes = _IMG_109.read_bytes()
  for i, fr in enumerate(images.AnimationFrames(b109, max_pixels=128, decimation=True)):
    p109 = pathlib.Path(_TEST_IMAGES_PATH / f'109-frame-{i:02d}.png')
    p109.write_bytes(fr)
    print(f'Wrote {p109} : {hashes.Hash256(p109.read_bytes()).hex()}')
  return 0


if __name__ == '__main__':
  raise SystemExit(Main())
