# Test Images

These are all public domain test images (pictures of old composers mostly), in `jpg`, `png`, and `gif` formats.

In particular:

- ![100.jpg](100.jpg) and ![105.jpg](105.jpg) are the exact same binary (image);
  - ![100-reduced-128.png](100-reduced-128.png) is the correct output of `transai.utils.images.ResizeImageForVision(open('100.jpg', 'rb').read(), max_pixels=128)`
- ![107.png](107.png) and ![108.png](108.png) are the same image, re-scaled (`108.jpg` is the smaller one);
- ![109.jpg](109.gif) is a long-ish animated GIF.
  - ![109-frame-00.png](109-frame-00.png) to ![109-frame-10.png](109-frame-10.png) are the correct 11 frames returned by `transai.utils.images.AnimationFrames(open('109.gif', 'rb').read(), max_pixels=128, decimation=True)`
