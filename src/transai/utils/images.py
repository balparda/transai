# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""AI image utilities."""

from __future__ import annotations

import collections.abc
import io
import logging
from typing import cast

from PIL import GifImagePlugin, Image, ImageSequence

from transai.core import ai

# we want to limit the image size for vision models, so we will start with 1024x1024 = 1 MP
_VISION_MAX_IMAGE_PIXELS: int = 1024  # max px dimension (width or height) for images sent to vision
# animations are reduced to ~10/11 frames of 336x336=0.108MP each x10/11 = ~1.08-1.18  MP
_VISION_MAX_ANIMATION_PIXELS: int = 336  # max px dimension for animated image frames sent to vision
_ANIMATION_FRAMES_TARGET: int = 10  # target number of frames to decimate to for animated images


def _ImageToScaledPNGBytes(
  img_obj: Image.Image, *, max_pixels: int = _VISION_MAX_IMAGE_PIXELS
) -> bytes:
  """Down-scale an image so its longest edge is at most `max_pixels`.

  If the image already fits it is returned unchanged (as PNG bytes).
  Animated formats (e.g. GIF) are reduced to the first frame.

  Args:
    img_obj: Pillow Image object
    max_pixels: maximum allowed pixel dimension for width or height

  Returns:
    PNG-encoded bytes of the (possibly resized) image

  """
  img_obj = img_obj.convert('RGB')  # drop alpha / palette / animation
  w: int
  h: int
  new_w: int
  new_h: int
  w, h = img_obj.size
  if max(w, h) > max_pixels:
    scale: float = max_pixels / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img_obj = img_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)
    logging.debug('Resized image from %dx%d to %dx%d for vision', w, h, new_w, new_h)
  buf: io.BytesIO = io.BytesIO()
  img_obj.save(buf, format='PNG')
  return buf.getvalue()


def ResizeImageForVision(
  image_bytes: bytes, *, max_pixels: int = _VISION_MAX_IMAGE_PIXELS
) -> bytes:
  """Down-scale an image so its longest edge is at most `max_pixels`.

  If the image already fits it is returned unchanged (as PNG bytes).
  Animated formats (e.g. GIF) are reduced to the first frame.

  Args:
    image_bytes: raw binary image data in any Pillow-supported format
    max_pixels: maximum allowed pixel dimension for width or height

  Returns:
    PNG-encoded bytes of the (possibly resized) image

  """
  return _ImageToScaledPNGBytes(Image.open(io.BytesIO(image_bytes)), max_pixels=max_pixels)


def AnimationFrames(
  img_bin: bytes,
  *,
  max_pixels: int = _VISION_MAX_ANIMATION_PIXELS,
  decimation: bool = True,
) -> collections.abc.Iterator[bytes]:
  """Convert an animated image to its individual frames, limited by at most `max_pixels`.

  Args:
    img_bin: bytes of the animated image to process
    max_pixels (default=_VISION_MAX_ANIMATION_PIXELS): maximum allowed animation pixel dimension
    decimation (default=True): whether to decimate frames, trying to aim for 10 frames;
        (I know this is not the real way decimation worked in the Roman Empire...)

  Yields:
    animation frames as individual PNG images, limited by `max_pixels` on the longest edge

  Raises:
    ai.Error: if the first frame fails to process or not an animation (less than 2 frames);
        for subsequent frames, errors are logged and ignored to allow partial results

  """
  frame_count: int = 0
  yield_count: int = 0
  n_frames: int = 0
  err_msg: str
  try:
    animation = cast('GifImagePlugin.GifImageFile', Image.open(io.BytesIO(img_bin)))
    if not animation.is_animated:
      raise ai.Error('Image is not animated GIF')
    n_frames = animation.n_frames
    if n_frames <= 1:
      raise ai.Error(f'Animation does not have multiple frames, got {n_frames}')
    decimate_factor: int = max(1, n_frames // _ANIMATION_FRAMES_TARGET)
    for frame_count, frame in enumerate(ImageSequence.Iterator(animation)):
      if decimation and frame_count % decimate_factor:
        continue
      try:
        yield _ImageToScaledPNGBytes(frame.copy(), max_pixels=max_pixels)
        yield_count += 1
      except OSError as err:
        err_msg = f'Error in animated frame {frame_count + 1}'
        if not frame_count:
          raise ai.Error(err_msg) from err  # this is the first image and shouldn't fail
        logging.warning(err_msg)  # the other frames can be logged and ignored
  except OSError as err:
    err_msg = f'Animation error @{frame_count + 1}'
    if not frame_count:
      raise ai.Error(err_msg) from err  # don't tolerate first-frame errors
    logging.warning(err_msg)  # but only log subsequent errors (in secondary frames)
  # check we did indeed have more than one frame, otherwise this was not an animation!
  if frame_count <= 1:
    raise ai.Error(f'Image is not animated, expected multiple frames, got {frame_count}')
  logging.debug(f'Resized animation with {n_frames} frames, yielded {yield_count} frames')
