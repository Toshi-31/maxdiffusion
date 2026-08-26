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
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from maxdiffusion.models.attention_flax import (
    _select_flash_block_sizes,
    _pad_data_for_flash,
)


class AttentionBlockSizesTest(unittest.TestCase):
  """Unit tests for Flash/Splash block-size selection and padding."""

  def test_symmetric_configured_behavior_unchanged(self):
    """Test A: User-configured symmetric BlockSizes are returned unchanged."""
    bs = splash_attention_kernel.BlockSizes(
        block_q=2048,
        block_kv=1024,
        block_kv_compute=512,
        block_q_dkv=1024,
        block_kv_dkv=512,
        block_kv_dkv_compute=256,
        block_q_dq=512,
        block_kv_dq=256,
        use_fused_bwd_kernel=False,
    )
    q = jnp.zeros((1, 1, 4096, 128), dtype=jnp.bfloat16)
    k = jnp.zeros((1, 1, 4096, 128), dtype=jnp.bfloat16)

    result = _select_flash_block_sizes(
        query=q,
        key=k,
        flash_block_sizes=bs,
        dtype=jnp.bfloat16,
        attention_kernel="flash",
        preserve_asymmetric_block_sizes=False,
    )
    self.assertIs(result, bs)

  def test_default_asymmetric_behavior_unchanged(self):
    """Test B: Default asymmetric behavior derives KV-aware BlockSizes matching main."""
    bs = splash_attention_kernel.BlockSizes(
        block_q=1024,
        block_kv=1024,
        block_kv_compute=1024,
        block_q_dkv=1024,
        block_kv_dkv=1024,
        block_kv_dkv_compute=1024,
        block_q_dq=1024,
        block_kv_dq=1024,
        use_fused_bwd_kernel=False,
    )
    q = jnp.zeros((1, 1, 512, 128), dtype=jnp.bfloat16)
    k = jnp.zeros((1, 1, 4096, 128), dtype=jnp.bfloat16)

    result = _select_flash_block_sizes(
        query=q,
        key=k,
        flash_block_sizes=bs,
        dtype=jnp.bfloat16,
        attention_kernel="flash",
        preserve_asymmetric_block_sizes=False,
    )
    self.assertEqual(result.block_q, 1024)
    self.assertEqual(result.block_kv, 4096)
    self.assertEqual(result.block_kv_compute, 4096)

  def test_klein_kv_asymmetric_config_preserved(self):
    """Test C: Klein KV asymmetric config is preserved when preserve_asymmetric_block_sizes=True."""
    klein_bs = splash_attention_kernel.BlockSizes(
        block_q=4608,
        block_kv=1024,
        block_kv_compute=1024,
        block_q_dkv=4608,
        block_kv_dkv=1024,
        block_kv_dkv_compute=1024,
        block_q_dq=4608,
        block_kv_dq=1024,
        use_fused_bwd_kernel=False,
    )
    q = jnp.zeros((1, 1, 4608, 128), dtype=jnp.bfloat16)
    k = jnp.zeros((1, 1, 8704, 128), dtype=jnp.bfloat16)

    result = _select_flash_block_sizes(
        query=q,
        key=k,
        flash_block_sizes=klein_bs,
        dtype=jnp.bfloat16,
        attention_kernel="flash",
        preserve_asymmetric_block_sizes=True,
    )
    self.assertIs(result, klein_bs)
    self.assertEqual(result.block_q, 4608)
    self.assertEqual(result.block_kv, 1024)
    self.assertEqual(result.block_kv_compute, 1024)
    self.assertNotEqual(8704 % 1024, 0)

  def test_pad_data_for_flash_validity(self):
    """Test D: _pad_data_for_flash pads 8704 KV tokens to 9216 (multiple of 1024)."""
    key = jnp.zeros((1, 8704, 8 * 128), dtype=jnp.bfloat16)
    padded_k, _, original_len = _pad_data_for_flash(
        key,
        heads=8,
        flash_block_size=1024,
    )
    self.assertEqual(original_len, 8704)
    self.assertEqual(padded_k.shape[2], 9216)
    self.assertEqual(padded_k.shape[2] % 1024, 0)


if __name__ == "__main__":
  unittest.main()
