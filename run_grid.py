import os
import subprocess
import re
import time

blocks_q = [512, 1024, 2048, 4096]
blocks_kv = [512, 1024, 2048, 4096]
blocks_kvc = [256, 512, 1024]

results = []

report_file = "workloads/reports/optimization_report_v1.md"
os.makedirs(os.path.dirname(report_file), exist_ok=True)

with open(report_file, "w") as f:
    f.write("# Block Size Grid Search Results\n\n")
    f.write("| Block Q | Block KV | Block KV Compute | Status | Decode Time (ms) |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- |\n")

def cleanup_tpu():
    subprocess.run("sudo fuser -k -9 /dev/vfio/0", shell=True, capture_output=True)
    subprocess.run("sudo pkill -9 -f benchmark_kl_wan.py", shell=True, capture_output=True)
    subprocess.run("sudo rm -f /tmp/libtpu_lockfile", shell=True, capture_output=True)

for bq in blocks_q:
    for bkv in blocks_kv:
        for bkvc in blocks_kvc:
            print(f"\n--- Testing Q={bq}, KV={bkv}, KVC={bkvc} ---")
            
            env = os.environ.copy()
            env["BLOCK_Q"] = str(bq)
            env["BLOCK_KV"] = str(bkv)
            env["BLOCK_KV_COMPUTE"] = str(bkvc)
            
            # Use XLA flags from previous experiment as a baseline for this grid search?
            # User wants to run standard benchmark, let's keep it simple without massive flags unless specified
            
            start_t = time.time()
            try:
                # We need to timeout in case JAX hangs
                process = subprocess.run(
                    ["python3", "benchmark_kl_wan.py"], 
                    env=env, 
                    cwd=None,
                    capture_output=True, 
                    text=True, 
                    timeout=300
                )
                
                output = process.stdout + process.stderr
                
                if process.returncode != 0:
                    status = "Failed"
                    if "VMEM" in output or "ResourceExhausted" in output:
                        status = "VMEM OOM"
                    time_val = "N/A"
                    print(f"Failed! Output snippet: {output[-500:]}")
                    cleanup_tpu()
                else:
                    match = re.search(r"Average VAE Decode Time .*?: ([\d\.]+) ms", output)
                    if match:
                        status = "Success"
                        time_val = match.group(1)
                        print(f"Success! Time: {time_val} ms")
                    else:
                        status = "Failed Parse"
                        time_val = "N/A"
                        print("Could not find decode time in output.")
            
            except subprocess.TimeoutExpired:
                status = "Timeout"
                time_val = "N/A"
                print("Process timed out.")
                cleanup_tpu()
            
            with open(report_file, "a") as f:
                f.write(f"| {bq} | {bkv} | {bkvc} | {status} | {time_val} |\n")
                
print("\nGrid search complete. Report saved to", report_file)
