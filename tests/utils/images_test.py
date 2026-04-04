# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""images.py unittest.

Run with:
  poetry run pytest -vvv tests/utils/images_test.py
"""

from __future__ import annotations

import io
import logging
import pathlib
from collections.abc import Iterator
from unittest import mock

import pytest
from PIL import Image, ImageFile, ImageSequence

from transai.core import ai
from transai.utils import images

_TEST_IMAGES_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / 'data' / 'images'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ReadTestImage(name: str, /) -> bytes:
  """Read a binary test image by filename from the shared test-images directory.

  Args:
    name: filename (e.g. '100.jpg') relative to _TEST_IMAGES_PATH.

  Returns:
    Raw bytes of the file.

  """
  return (_TEST_IMAGES_PATH / name).read_bytes()


def _MakeAnimatedGif(n_frames: int, /, *, width: int = 20, height: int = 20) -> bytes:
  """Create a minimal in-memory animated GIF with ``n_frames`` distinct frames.

  Each frame has a different shade of red so that Pillow records them as
  individual frames.  At least 2 frames are required.

  Args:
    n_frames: number of frames to include (must be >= 2).
    width: frame width in pixels.
    height: frame height in pixels.

  Returns:
    GIF-encoded bytes of the animated image.

  Raises:
    ValueError: if n_frames < 2.

  """
  if n_frames < 2:
    raise ValueError(f'n_frames must be >= 2, got {n_frames}')
  step: int = 255 // (n_frames - 1) if n_frames > 1 else 255
  frames: list[Image.Image] = [
    Image.new('RGB', (width, height), color=(i * step, 0, 0)) for i in range(n_frames)
  ]
  buf = io.BytesIO()
  frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=100)
  return buf.getvalue()


def _PngSize(png_bytes: bytes, /) -> tuple[int, int]:
  """Return (width, height) of a PNG from its raw bytes.

  Args:
    png_bytes: raw PNG-encoded image bytes.

  Returns:
    Tuple (width, height).

  """
  return Image.open(io.BytesIO(png_bytes)).size


# ---------------------------------------------------------------------------
# ResizeImageForVision
# ---------------------------------------------------------------------------


def test_resize_image_for_vision_small_image_not_resized() -> None:
  """100.jpg (200x246) with max_pixels=512 should not be resized (246 < 512).

  Also verifies the output is PNG-encoded.
  """
  result: bytes = images.ResizeImageForVision(_ReadTestImage('100.jpg'), max_pixels=512)
  w, h = _PngSize(result)
  assert (w, h) == (200, 246)
  # PNG header magic bytes
  assert result[:4] == b'\x89PNG'


def test_resize_image_for_vision_tall_image_exact_bytes() -> None:
  """100.jpg (200x246) resized to max_pixels=128 must match the precomputed reference exactly."""
  result: bytes = images.ResizeImageForVision(_ReadTestImage('100.jpg'), max_pixels=128)
  reference: bytes = _ReadTestImage('100-reduced-128.png')
  assert result == reference


def test_resize_image_for_vision_tall_image_dimension_check() -> None:
  """100.jpg (200x246): after resize to max_pixels=128, height ≤ 128 and is the dominant edge."""
  result: bytes = images.ResizeImageForVision(_ReadTestImage('100.jpg'), max_pixels=128)
  w, h = _PngSize(result)
  assert h == 128  # tall side becomes exactly max_pixels
  assert w < 128  # narrow side is proportionally smaller


def test_resize_image_for_vision_wide_image_width_dominant() -> None:
  """106.jpg (300x222) with max_pixels=200: width is dominant, should be scaled to 200."""
  result: bytes = images.ResizeImageForVision(_ReadTestImage('106.jpg'), max_pixels=200)
  w, h = _PngSize(result)
  assert w == 200  # wide side becomes exactly max_pixels
  assert h < 200  # narrow side is proportionally smaller
  assert result[:4] == b'\x89PNG'


def test_resize_image_for_vision_rgba_converted_to_rgb() -> None:
  """107.png is RGBA (170x225); the output must be an RGB PNG regardless of input mode."""
  result: bytes = images.ResizeImageForVision(_ReadTestImage('107.png'), max_pixels=512)
  out: ImageFile.ImageFile = Image.open(io.BytesIO(result))
  assert out.format == 'PNG'
  assert out.mode == 'RGB'
  assert out.size == (170, 225)  # smaller than 512 → no resize


def test_resize_image_for_vision_default_max_pixels_leaves_small_images_unchanged() -> None:
  """All test images are smaller than the default 1024-px limit: size must be unchanged."""
  for name in ('100.jpg', '101.jpg', '102.jpg', '103.jpg', '104.png'):
    orig_size: tuple[int, int] = Image.open(io.BytesIO(_ReadTestImage(name))).size
    result: bytes = images.ResizeImageForVision(_ReadTestImage(name))
    assert _PngSize(result) == orig_size, f'{name}: size changed unexpectedly'


def test_resize_image_for_vision_duplicate_images_give_identical_results() -> None:
  """100.jpg and 105.jpg are the same binary: both must produce the same PNG output."""
  result_100: bytes = images.ResizeImageForVision(_ReadTestImage('100.jpg'), max_pixels=128)
  result_105: bytes = images.ResizeImageForVision(_ReadTestImage('105.jpg'), max_pixels=128)
  assert result_100 == result_105


def test_resize_image_for_vision_all_formats_output_png() -> None:
  """JPEG and PNG inputs should all produce valid PNG output."""
  for name in ('100.jpg', '104.png', '107.png', '108.png'):
    result: bytes = images.ResizeImageForVision(_ReadTestImage(name), max_pixels=512)
    assert result[:4] == b'\x89PNG', f'{name}: expected PNG magic bytes'


def test_resize_image_for_vision_small_max_pixels_produces_min_one_pixel() -> None:
  """With max_pixels=1, a rectangular image must produce at least a 1x1 PNG (no zero-pixel dims)."""
  # Create a synthetic 100x200 image in-memory
  buf = io.BytesIO()
  Image.new('RGB', (100, 200), color=(128, 64, 32)).save(buf, format='JPEG')
  result: bytes = images.ResizeImageForVision(buf.getvalue(), max_pixels=1)
  w, h = _PngSize(result)
  assert w >= 1
  assert h >= 1


# ---------------------------------------------------------------------------
# AnimationFrames
# ---------------------------------------------------------------------------


def _MakeSingleFrameGif() -> bytes:
  """Create a minimal 1-frame (non-animated) GIF in memory.

  Pillow reports is_animated=False for single-frame GIFs, which is the
  correct trigger for the 'Image is not animated GIF' error path.

  Returns:
    GIF-encoded bytes of a 1-frame image.

  """
  buf = io.BytesIO()
  Image.new('RGB', (10, 10), color=(100, 0, 0)).save(buf, format='GIF')
  return buf.getvalue()


def test_animation_frames_single_frame_gif_raises_not_animated() -> None:
  """A 1-frame GIF (is_animated=False) must raise ai.Error 'not animated GIF'.

  Exercises the `if not animation.is_animated:` True branch.
  """
  with pytest.raises(ai.Error, match='Image is not animated GIF'):
    list(images.AnimationFrames(_MakeSingleFrameGif()))


def test_animation_frames_real_gif_with_decimation_matches_reference_frames() -> None:
  """109.gif + decimation=True + max_pixels=128 must produce the 11 precomputed reference frames.

  This test exercises: the decimation-skip branch, frame resize, and the happy-path
  debug log (frame_count=118 > 1 at the end).
  """
  gif_bytes: bytes = _ReadTestImage('109.gif')
  reference: list[bytes] = [_ReadTestImage(f'109-frame-{i:02d}.png') for i in range(11)]
  produced: list[bytes] = list(images.AnimationFrames(gif_bytes, max_pixels=128, decimation=True))
  assert len(produced) == 11
  for i, (ref, prod) in enumerate(zip(reference, produced, strict=True)):
    assert ref == prod, f'frame {i:02d} bytes do not match reference'


def test_animation_frames_real_gif_without_decimation_yields_all_frames() -> None:
  """109.gif + decimation=False must yield all 119 frames, each a valid PNG."""
  gif_bytes: bytes = _ReadTestImage('109.gif')
  produced: list[bytes] = list(images.AnimationFrames(gif_bytes, decimation=False))
  assert len(produced) == 119
  # Spot-check first and last frame are valid PNG
  for idx in (0, 118):
    assert produced[idx][:4] == b'\x89PNG', f'frame {idx} is not a PNG'


def test_animation_frames_two_frame_gif_raises_insufficient_frame_count() -> None:
  """A 2-frame GIF ends with frame_count=1 (≤1), so ai.Error must be raised.

  This exercises the `if frame_count <= 1:` True branch.
  """
  gif_bytes: bytes = _MakeAnimatedGif(2)
  with pytest.raises(ai.Error, match='not animated, expected multiple frames'):
    list(images.AnimationFrames(gif_bytes, decimation=False))


def test_animation_frames_three_frame_gif_completes_successfully() -> None:
  """A 3-frame GIF ends with frame_count=2 (>1): no error, 3 PNG frames produced.

  This exercises the `if frame_count <= 1:` False branch and the final debug log.
  """
  gif_bytes: bytes = _MakeAnimatedGif(3)
  produced: list[bytes] = list(images.AnimationFrames(gif_bytes, decimation=False))
  assert len(produced) == 3
  for i, frame_bytes in enumerate(produced):
    assert frame_bytes[:4] == b'\x89PNG', f'frame {i} is not a PNG'


def test_animation_frames_custom_gif_decimation_reduces_frame_count() -> None:
  """21-frame GIF + decimation=True: decimate_factor=2, so 11 frames are yielded (0,2,4,…,20).

  This exercises the decimation `continue` branch (frame_count % decimate_factor != 0).
  """
  gif_bytes: bytes = _MakeAnimatedGif(21)
  produced: list[bytes] = list(images.AnimationFrames(gif_bytes, decimation=True))
  assert len(produced) == 11


def test_animation_frames_mock_n_frames_one_raises() -> None:
  """Mock where is_animated=True but n_frames=1 must raise ai.Error.

  This exercises the `if n_frames <= 1:` branch which cannot be triggered with a
  real GIF (Pillow only marks is_animated=True when n_frames > 1).
  """
  mock_anim = mock.MagicMock()
  mock_anim.is_animated = True
  mock_anim.n_frames = 1
  with (
    mock.patch.object(Image, 'open', return_value=mock_anim),
    pytest.raises(ai.Error, match='does not have multiple frames'),
  ):
    list(images.AnimationFrames(b'fake gif bytes'))


def test_animation_frames_open_oserror_raises_ai_error() -> None:
  """OSError from Image.open (frame_count=0) must be re-raised as ai.Error.

  This exercises the outer `except OSError` with `if not frame_count:` True branch.
  """
  with (
    mock.patch.object(Image, 'open', side_effect=OSError('bad file')),
    pytest.raises(ai.Error, match='Animation error'),
  ):
    list(images.AnimationFrames(b'corrupted data'))


def test_animation_frames_inner_oserror_first_frame_raises() -> None:
  """OSError inside _ImageToScaledPNGBytes on the first frame must raise ai.Error.

  This exercises the inner `except OSError` with `if not frame_count:` True branch.
  """
  gif_bytes: bytes = _MakeAnimatedGif(5)
  with (
    mock.patch(
      'transai.utils.images._ImageToScaledPNGBytes', side_effect=OSError('bad frame data')
    ),
    pytest.raises(ai.Error, match='Error in animated frame'),
  ):
    list(images.AnimationFrames(gif_bytes, decimation=False))


def test_animation_frames_inner_oserror_later_frame_logged_and_skipped(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """OSError inside _ImageToScaledPNGBytes on frame 1 must be logged; other frames still yielded.

  This exercises the inner `except OSError` with `if not frame_count:` False branch.
  """
  gif_bytes: bytes = _MakeAnimatedGif(5)
  call_count = 0

  def _side_effect(*_args: object, **_kwargs: object) -> bytes:
    nonlocal call_count
    call_count += 1
    if call_count == 2:  # second call (frame_count=1) raises
      raise OSError('bad frame 1')
    return b'fake_png_bytes'

  with (
    mock.patch('transai.utils.images._ImageToScaledPNGBytes', side_effect=_side_effect),
    caplog.at_level(logging.ERROR),
  ):
    produced: list[bytes] = list(images.AnimationFrames(gif_bytes, decimation=False))

  # 5 frames attempted, 1 failed → 4 yielded
  assert len(produced) == 4
  assert any('Error in animated frame' in r.message for r in caplog.records)


def test_animation_frames_outer_oserror_after_frames_logged(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """OSError raised by the iterator itself (after 3 successful frames) must be logged.

  frame_count=2 when the error fires → outer `except OSError`, `if not frame_count:` False
  branch → logging.exception, then `if frame_count <= 1:` False → logging.debug.
  This is the only test that exercises line 117 (outer except logging.exception).
  """
  frame_img = Image.new('RGB', (5, 5), color=(100, 100, 100))

  def _bad_iterator(_anim: object, /) -> Iterator[Image.Image]:
    yield frame_img  # frame_count=0
    yield frame_img  # frame_count=1
    yield frame_img  # frame_count=2
    raise OSError('corrupt mid-iteration')  # fires on the 4th __next__ call

  mock_anim = mock.MagicMock()
  mock_anim.is_animated = True
  mock_anim.n_frames = 10

  with (
    mock.patch.object(Image, 'open', return_value=mock_anim),
    mock.patch.object(ImageSequence, 'Iterator', side_effect=_bad_iterator),
    caplog.at_level(logging.ERROR),
  ):
    produced: list[bytes] = list(images.AnimationFrames(b'fake', decimation=False))

  assert len(produced) == 3  # 3 frames yielded before the iterator failed
  assert any('Animation error' in r.message for r in caplog.records)


def test_animation_frames_max_pixels_parameter_controls_output_size() -> None:
  """Frames produced with a small max_pixels must fit within that pixel budget."""
  gif_bytes: bytes = _ReadTestImage('109.gif')  # 500x100 frames, so resize is always required
  produced: list[bytes] = list(images.AnimationFrames(gif_bytes, max_pixels=64, decimation=True))
  assert produced  # at least one frame
  for i, frame_bytes in enumerate(produced):
    w, h = _PngSize(frame_bytes)
    assert max(w, h) <= 64, f'frame {i} exceeds max_pixels=64: {w}x{h}'
