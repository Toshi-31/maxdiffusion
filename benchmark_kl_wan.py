import time
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
import numpy as np

def run_benchmark():
    devices = jax.devices()
    device_array = np.array(devices[:4]).reshape((1, 4))
    mesh = Mesh(device_array, ('redundant', 'vae_spatial'))

    rngs = nnx.Rngs(0)
    
    batch, time_dim, height, width, z_dim = 1, 1, 135, 240, 16 
    dummy_z = jnp.ones((batch, time_dim, height, width, z_dim), dtype=jnp.bfloat16)
    
    spatial_sharding = NamedSharding(mesh, P(None, None, None, "vae_spatial", None))
    dummy_z = jax.device_put(dummy_z, spatial_sharding)

    # Targeted list of untested (bq, bk) combinations
    block_sizes = [
        (128, 1024), (256, 1024), (512, 1024),
        (1024, 1024),
        (2048, 128), (2048, 256), (2048, 512), (2048, 1024), (2048, 2048)
    ]
    
    with jax.set_mesh(mesh):
        for bq, bk in block_sizes:
            print(f"--- Testing block_q={bq}, block_k={bk} ---")
            
            vae = AutoencoderKLWan(rngs=rngs, mesh=mesh, dtype=jnp.bfloat16, flash_block_q=bq, flash_block_k=bk)
            cache = AutoencoderKLWanCache(vae)
            
            def decode_step(z, feat_cache):
                return vae.decode(z, feat_cache)

            print("JIT Compiling...")
            out = decode_step(dummy_z, cache)
            jax.block_until_ready(out)

            print("Running Benchmark...")
            start = time.perf_counter()
            
            iters = 5
            for _ in range(iters):
                out = decode_step(dummy_z, cache)
            jax.block_until_ready(out)
            
            end = time.perf_counter()
            
            print(f"Average VAE Decode Time [q={bq}, k={bk}]: {(end - start) / iters * 1000:.2f} ms\n")

if __name__ == "__main__":
    run_benchmark()
