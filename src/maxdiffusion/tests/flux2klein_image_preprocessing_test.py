"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
import numpy as np
import pytest
from PIL import Image

from maxdiffusion.pipelines.flux.flux2klein_pipeline import FlaxFlux2KleinPipeline


class TestFlux2KleinImagePreprocessing(unittest.TestCase):
  """Unit test suite for canonical reference image preprocessing in FLUX.2-Klein."""

  def test_a_no_duplicate_resize(self):
    """Test A: Verifies that passing a raw PIL image is preprocessed cleanly without pre-alteration."""
    img = Image.new("RGB", (512, 512), color=(120, 150, 200))
    res = FlaxFlux2KleinPipeline.preprocess_reference_image(img)
    self.assertEqual(res.shape, (1, 3, 512, 512))

  def test_b_aspect_ratio_preserved(self):
    """Test B: Rectangular images must preserve aspect ratio and not be forced into a square or output shape."""
    # 1600 x 900 -> total area 1,440,000 > 1024^2 (1,048,576)
    img = Image.new("RGB", (1600, 900), color=(50, 100, 150))
    res = FlaxFlux2KleinPipeline.preprocess_reference_image(img)
    _, _, h, w = res.shape
    self.assertEqual(w % 16, 0)
    self.assertEqual(h % 16, 0)
    self.assertAlmostEqual(w / h, 1600 / 900, delta=0.1)
    self.assertLessEqual(w * h, 1024 * 1024 + 16 * 1024)

  def test_c_output_size_independence(self):
    """Test C: Preprocessed reference image must be independent of generation output dimensions."""
    img = Image.new("RGB", (640, 480), color=(200, 100, 50))
    res1 = FlaxFlux2KleinPipeline.preprocess_reference_image(img)
    res2 = FlaxFlux2KleinPipeline.preprocess_reference_image(img)
    np.testing.assert_array_equal(res1, res2)
    self.assertEqual(res1.shape, (1, 3, 480, 640))

  def test_d_vae_ready_format(self):
    """Test D: Verifies output shape is [1, 3, H, W], H%16==0, W%16==0, and values are in [-1, 1]."""
    img = Image.new("RGB", (345, 678), color=(255, 128, 0))
    res = FlaxFlux2KleinPipeline.preprocess_reference_image(img)
    self.assertEqual(res.ndim, 4)
    self.assertEqual(res.shape[0], 1)
    self.assertEqual(res.shape[1], 3)
    self.assertEqual(res.shape[2] % 16, 0)
    self.assertEqual(res.shape[3] % 16, 0)
    self.assertTrue(np.all(res >= -1.0) and np.all(res <= 1.0))
    self.assertEqual(res.dtype, np.float32)

  def test_e_input_validation(self):
    """Test E: Verifies clear error on unsupported types, small images, and extreme aspect ratios."""
    # Non-PIL input
    with self.assertRaises(TypeError):
      FlaxFlux2KleinPipeline.preprocess_reference_image(np.zeros((512, 512, 3), dtype=np.uint8))

    with self.assertRaises(TypeError):
      FlaxFlux2KleinPipeline.preprocess_reference_image("path/to/image.png")

    # Too small (min side < 64)
    small_img = Image.new("RGB", (32, 128))
    with self.assertRaises(ValueError):
      FlaxFlux2KleinPipeline.preprocess_reference_image(small_img)

    # Extreme aspect ratio (> 8:1)
    extreme_img = Image.new("RGB", (1000, 100))
    with self.assertRaises(ValueError):
      FlaxFlux2KleinPipeline.preprocess_reference_image(extreme_img)

  def test_f_diffusers_exact_parity(self):
    """Test F: Direct numerical parity comparison with Diffusers Flux2ImageProcessor."""
    try:
      from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    except ImportError:
      pytest.skip("diffusers not installed or Flux2ImageProcessor not available")

    proc = Flux2ImageProcessor()

    # Test images of various shapes and content
    test_dims = [(512, 512), (768, 512), (1200, 800), (1600, 900)]
    for w, h in test_dims:
      # Generate non-trivial image with color gradient
      x = np.linspace(0, 255, w, dtype=np.uint8)
      y = np.linspace(0, 255, h, dtype=np.uint8)
      xx, yy = np.meshgrid(x, y)
      arr = np.stack([xx, yy, ((xx + yy) // 2).astype(np.uint8)], axis=-1)
      pil_img = Image.fromarray(arr)

      # 1. Diffusers preprocessing
      image_w, image_height = pil_img.size
      if image_w * image_height > 1024 * 1024:
        diffusers_pil = proc._resize_to_target_area(pil_img, 1024 * 1024)
        image_w, image_height = diffusers_pil.size
      else:
        diffusers_pil = pil_img
      image_w = (image_w // 16) * 16
      image_height = (image_height // 16) * 16
      diffusers_tensor = proc.preprocess(diffusers_pil, height=image_height, width=image_w, resize_mode="crop")
      diffusers_np = diffusers_tensor.detach().cpu().float().numpy()

      # 2. MaxDiffusion canonical preprocessing
      maxdiff_np = FlaxFlux2KleinPipeline.preprocess_reference_image(pil_img)

      # Assert identical shape
      self.assertEqual(diffusers_np.shape, maxdiff_np.shape)

      # Assert numerical equivalence
      mae = np.mean(np.abs(diffusers_np - maxdiff_np))
      max_diff = np.max(np.abs(diffusers_np - maxdiff_np))
      self.assertLess(mae, 1e-4, f"MAE {mae} exceeded threshold for dims ({w}, {h})")
      self.assertLess(max_diff, 1e-3, f"Max diff {max_diff} exceeded threshold for dims ({w}, {h})")


if __name__ == "__main__":
  unittest.main()
