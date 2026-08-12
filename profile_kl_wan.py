import time
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
import numpy as np

def run_profiler():
    devices = jax.devices()
    device_array = np.array(devices[:4]).reshape((1, 4))
    mesh = Mesh(device_array, ('redundant', 'vae_spatial'))

    rngs = nnx.Rngs(0)
    
    batch, time_dim, height, width, z_dim = 1, 1, 135, 240, 16 
    dummy_z = jnp.ones((batch, time_dim, height, width, z_dim), dtype=jnp.bfloat16)
    
    spatial_sharding = NamedSharding(mesh, P(None, None, None, "vae_spatial", None))
    dummy_z = jax.device_put(dummy_z, spatial_sharding)

    # Use the best tuned block sizes we found!
    bq, bk = 1024, 1024
    
    with jax.set_mesh(mesh):
        print(f"--- Profiling block_q={bq}, block_k={bk} ---")
        
        vae = AutoencoderKLWan(rngs=rngs, mesh=mesh, dtype=jnp.bfloat16, flash_block_q=bq, flash_block_k=bk)
        cache = AutoencoderKLWanCache(vae)
        
        def decode_step(z, feat_cache):
            return vae.decode(z, feat_cache)

        print("JIT Compiling and Warmup...")
        out = decode_step(dummy_z, cache)
        jax.block_until_ready(out)

        print("Running XProf Trace...")
        # Start profiler
        with jax.profiler.trace("/tmp/xprof_logs", create_perfetto_link=False):
            out = decode_step(dummy_z, cache)
            jax.block_until_ready(out)
        
        print("\nSuccess! Profiling trace saved to: /tmp/xprof_logs")
        print("You can zip this directory and open it with TensorBoard.")

if __name__ == "__main__":
    run_profiler()
