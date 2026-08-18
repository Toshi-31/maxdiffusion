---
name: tpu-optimization-workflow
description: >-
  Master loop for optimizing JAX/Flax or PyTorch-via-torchax models on Cloud TPUs: set up a measurable experiment, profile, form one hypothesis, change one thing, re-profile, keep or revert. Use when asked to make a TPU workload faster, to plan an optimization campaign, or when you have a slow model and do not yet know where to start. Triggers: "optimize this on TPU", "model is slow", "improve step time", "where do I start".
---

# TPU Model Optimization Workflow
This document is the master loop for optimizing JAX/Flax models and PyTorch models running through torchax on Cloud TPUs. Use it to decide what to inspect, what to profile, which skill to open, and when to keep or discard an experiment.

The workflow is intentionally iterative:

1. Add profiler code or trace markers around the model path being optimized.
2. Inspect the code for issues visible without a trace.
3. Capture a short steady-state trace, usually around 5 representative steps.
4. Analyze the trace and state a clear hypothesis.
5. Try one fix at a time.
6. Re-profile the same scenario.
7. If the fix works, keep it and checkpoint it with source control. If it does not work, revert only that experiment's edits, update the hypothesis, and continue.
8. Repeat until the profile is near the relevant ceiling, the remaining bottleneck is outside the current scope, or there are no credible options left. When several diverging fixes look plausible, ask the user which direction to try first. Always ask the user before quantizing beyond BF16 or adding a new custom Pallas kernel.

After any obvious reshape, transpose, copy, stack, or concat issue is spotted, the most rewarding early paths are usually attention, FlashAttention block-size tuning, and XLA flag tuning for exposed collectives or scheduling gaps.

## 0. Set Up the Experiment
Goal: make every optimization measurable and reversible.

Before changing performance behavior:

* Identify the workload: training, inference, serving, diffusion sampling, VAE decode, attention-heavy generation, or embedding/retrieval.
* Identify the framework path: JAX/Flax, Flax NNX, PyTorch via torchax, or mixed.
* Record hardware and run config: TPU generation, slice size, single-slice vs. multislice, dtype, batch size, sequence/video length, mesh shape, logical_axis_rules, attention kernel, and XLA flags.
* Choose one success metric: step time, tokens/sec, images/sec, videos/sec, compile time, TPU idle percentage, collective percentage, MXU duty cycle, HBM bandwidth, or end-to-end latency.
* Check source control status before editing. Keep unrelated user changes intact. Use commits, branches, or named patches to checkpoint successful experiments when appropriate.
* Use `skills/profiling/README.md` to add profiler collection and trace markers. If the run does not already produce a useful profile, add the smallest amount of profiling code needed to capture the target region.

## 1. Inspect Before Profiling
Goal: catch obvious issues without overfitting to trace noise.

Read the model, input pipeline, sharding setup, attention path, dtype policy, and launch code. Look for:

* Python loops in hot JAX paths where scan, vectorization, or batching is expected.
* Repeated JIT construction, dynamic shapes, non-static arguments, or host syncs.
* Torch/JAX boundary movement, torchax fallback, mutable state hidden inside compiled code, or parameters not moved to JAX device arrays.
* Accidental Float64, unnecessary FP32, or suspicious convert churn.
* Manual device reshapes instead of topology-aware mesh construction.
* Logical-axis rules that do not match the tensor shapes they annotate.
* Tensor parallelism enabled by default without a memory or profile reason.
* Attention kernel choices that do not match sequence length, head divisibility, or mesh layout.

Give obvious reshape/layout issues a quick pass before deeper tuning. If the code is clearly materializing reshapes, transposes, copies, stacks, or concats in the hot path, fix those first because they can hide the real profile signal.

Route visible issues through:

| Visible Issue | Skill |
| :--- | :--- |
| JAX compile, host sync, loops, input transfer, or NNX state issues. | `skills/jax/jax_efficiency.md` |
| PyTorch via torchax movement, fallback, state, or sharding mechanics. | `skills/torchax/torchax_efficiency.md` |
| Dtype policy or Float64/FP32 mistakes. | `skills/performance/precision.md` |
| Physical TPU mesh construction or ICI/DCN axis placement. | `skills/hardware/topology.md` |
| Logical sharding, layout, or collective-heavy annotations. | `skills/performance/sharding_and_layout.md` |
| Attention kernel or attention parallelism mismatch. | `skills/performance/attention.md` |

If a code issue is obvious and low-risk, fix it, then still profile to confirm.

## 2. Capture a Short Trace
Goal: collect enough evidence to form a precise hypothesis.

Capture a short steady-state trace, usually about 5 steps. Avoid including startup, compilation, checkpoint load, dataset warmup, or first-token effects unless those are the target bottleneck.

Start with `skills/profiling/README.md`:

| Tool | Use |
| :--- | :--- |
| `get_overview` | Utilization, compilation time, TPU idle, compute/memory split, and high-level hot spots. |
| `get_top_hlo_ops` | Hottest HLO ops, collective/data-formatting time, source locations, FLOPs, and bandwidth. |
| `get_graph_viewer` | Producer/consumer neighborhoods for suspicious HLO ops. |
| `get_trace_window` | Timeline windows, device busy/idle, and large gaps. |
| `get_op_trace` | Kernel expansion for a named op or module. |
| `get_roofline` | Distance from compute, HBM, and communication ceilings. |

Use `skills/roofline_analysis/README.md` when the profile does not make the bottleneck class obvious.

## 3. Analyze and State the Hypothesis
Goal: make the next edit explainable before making it.

For each trace, write down:

1. The bottleneck class: host idle, recompilation, dtype/layout overhead, collective communication, attention, HBM bandwidth, MXU compute, or exposed scheduler/overlap issue.
2. The evidence: exact hot op names, timeline region, module name, collective type, layout conversion, roofline result, or source location.
3. The suspected cause in code or config.
4. The one proposed fix.
5. The expected profile change after the fix.

Use roofline analysis when forming the hypothesis, especially when the trace only says "this module is slow" without explaining why:

| Roofline Signal | What It Means | Usual Next Move |
| :--- | :--- | :--- |
| Memory-bound below the HBM ceiling | The model is wasting memory bandwidth through layout glue, materialization, or poor fusion. | Check reshapes/copies/transposes, sharding/layout rules, fusion barriers, and attention block sizes. |
| Memory-bound on the HBM ceiling | The kernel is genuinely bandwidth-limited. | Improve reuse with attention/block-size tuning, reduce avoidable reads/writes, or stop if it is near the ceiling. |
| Compute-bound below peak | The kernel is not feeding the MXU efficiently. | Check precision policy, GEMM shapes, fusion barriers, and attention kernel choice. |
| Communication-bound | Collectives dominate the step or block compute overlap. | Check logical sharding, physical topology, and then XLA overlap/latency-hiding flags. |
| Near the relevant ceiling | The module is already well optimized. | Move to the next bottleneck instead of over-tuning. |

State hypotheses on the fly as: "The profile says X is slow because it is memory/compute/communication-bound; therefore the next single fix is Y, and I expect metric Z to move."

After obvious reshape/layout problems are handled or ruled out, bias the next hypotheses toward the high-reward TPU levers:

1. Attention kernel and attention parallelism.
2. FlashAttention/Pallas block-size tuning.
3. XLA flag tuning for exposed collectives, latency hiding, and VMEM pressure.

Routing from profile evidence:

| Profile Signal | Next Tool | Skill |
| :--- | :--- | :--- |
| TPU idle gaps, input stalls, repeated host work, or compilation. | `get_trace_window`, `get_overview` | `skills/jax/jax_efficiency.md` or `skills/torchax/torchax_efficiency.md` |
| Float64 ops, FP32-heavy matmuls, excessive convert, or dtype churn. | `get_top_hlo_ops`, `get_graph_viewer` | `skills/performance/precision.md` |
| copy, transpose, reshape, pad, slice, or unfused elementwise chains. | `get_top_hlo_ops`, `get_graph_viewer` | `skills/performance/fusion.md`, `skills/performance/sharding_and_layout.md` |
| All-Gather, All-Reduce, Reduce-Scatter, All-to-All, or collective-permute. | `get_top_hlo_ops`, `get_graph_viewer` | `skills/performance/sharding_and_layout.md`, `skills/hardware/topology.md` |
| Hot attention custom calls, attention OOMs, or sequence-scaling issues. | `get_top_hlo_ops`, `get_roofline` | `skills/performance/attention.md` |
| Low MXU duty cycle or unclear compute vs. memory bottleneck. | `get_roofline` | `skills/roofline_analysis/README.md` |
| Exposed collectives remain after sharding/layout fixes. | `get_trace_window`, `get_top_hlo_ops` | `skills/performance/xla_flags.md` |

If the evidence supports several unrelated hypotheses, stop and ask the user to choose a priority, such as lower latency, higher throughput, lower memory, or least invasive code change.

## 4. Try One Fix
Goal: change one variable so the next profile is interpretable.

Keep experiments narrow:

* Fix host/runtime issues before model-graph tuning.
* Apply lossless dtype fixes before compiler flags or sharding experiments.
* Fix obvious reshape, transpose, copy, stack, or concat issues before attention and XLA tuning when they are visible in code or trace.
* Build the physical mesh with `skills/hardware/topology.md` before changing logical sharding.
* Use `skills/performance/sharding_and_layout.md` for DP/FSDP/CP/TP choices, logical-axis rules, and layout mismatch fixes.
* Use `skills/performance/attention.md` for attention kernel selection, Ulysses/ring/local choices, and Pallas block tuning. This is one of the highest-reward early paths for attention-heavy models.
* Ask the user before implementing a new custom Pallas kernel. Tuning an existing Pallas/FlashAttention block size is part of the normal loop; writing a new kernel is a larger design step. For guidelines on writing and optimizing distributed Pallas kernels, refer to `skills/kernels/SKILL.md`.
* Use `skills/performance/xla_flags.md` only after code, dtype, mesh, and sharding are reasonable. When exposed collectives or scheduling gaps remain, XLA flag tuning is a high-reward next experiment.
* Use `skills/performance/fusion.md` when the trace proves fusion barriers or kernel-level layout glue remain.

For source control:

* Check git diff before and after the experiment.
* Keep unrelated dirty files untouched.
* If the experiment succeeds, commit or otherwise checkpoint the focused diff when the user wants durable history.
* If the experiment fails, revert only the edits from that experiment. Do not revert unrelated user changes.

For precision:

* BF16 and FP32 policy fixes are normal lossless optimization work.
* Ask the user before quantizing beyond BF16 or introducing any quality-sensitive lower-precision path.

## 5. Re-Profile and Decide
Goal: keep only changes that improve the measured target or unlock the next credible improvement.

Run the same scenario again, preferably with the same trace window and number of steps. Compare:

* Target metric before vs. after.
* Hot op ranking and total step time.
* TPU idle percentage.
* Collective percentage and exposed collective duration.
* MXU duty cycle and HBM bandwidth.
* Any new regressions, OOM risk, numerical risk, or code complexity.

Decision rules:

| Result | Action |
| :--- | :--- |
| Clear improvement, no unacceptable regression. | Keep it, record evidence, and checkpoint with source control if appropriate. |
| No measurable change. | Revert the experiment, update the hypothesis, and pick the next highest-evidence option. |
| Mixed result. | Ask the user if the trade-off is acceptable, or run the smallest extra profile needed to decide. |
| Regression or instability. | Revert the experiment and document why the hypothesis failed. |
| Profile is near roofline or remaining bottleneck is external. | Stop tuning that path and summarize remaining limits. |

Then loop back to code inspection and trace analysis.

## 6. Report and Visualize
Goal: present the optimization results and code changes clearly in the Jetski UI.

When reporting results to the user:

* Do NOT Overwrite Baseline: Always version your files (v2, v3, etc.) for models, runner scripts, and logs so the history is preserved in the workspace.
* Write Structured Reports: Create `workloads/reports/optimization_report_v{N}.md` with a detailed comparison table mapping:
  * Average Step Time
  * Accumulated and Incremental Speedup
  * Collective Communication Volume (MB)
  * Number of GEMM Kernels per layer
  * Achieved FLOP Rate and Peak HW Utilization %
* Use Generative UI for Diffs: Whenever you create a new optimized model file, use the generative_ui skill to write a self-contained `code_diffs.html` in the brain directory. Embed it inline using `<agent-embed>` to show a GitHub-style side-by-side split diff highlighting the changes in red (removals) and green (additions).
* Perform Roofline Calculations: Always calculate the achieved FLOP rate against the physical peak of the TPU target (e.g., 1752 TFLOP/s for v6e-8) to verify if the exit criteria is met.

## Stop Conditions
Stop the optimization loop when one of these is true:

* The selected metric has reached the user's target.
* Roofline analysis shows the hot module is close to the relevant hardware ceiling.
* Remaining bottlenecks are input data, external service latency, unsupported kernels, or product constraints outside the current scope.
* Further improvements require quality-sensitive or larger architectural changes that the user has not approved.
* Further improvements require quantization beyond BF16 or a new custom Pallas kernel and the user has not approved that direction.
* You have multiple plausible but divergent paths and need user preference to continue responsibly.

## Model Optimization Checklist
* [ ] Profiling code or trace markers cover the target model path.
* [ ] Baseline run config and metric are recorded.
* [ ] Source control status is known before edits.
* [ ] Code inspection found and addressed obvious issues.
* [ ] Obvious reshape, transpose, copy, stack, or concat issues are fixed or ruled out.
* [ ] A short steady-state trace, usually around 5 steps, has been captured.
* [ ] The hypothesis names the hot op/module, evidence, cause, fix, and expected profile change.
* [ ] Roofline analysis is used when needed to explain whether the bottleneck is memory-bound, compute-bound, communication-bound, or near ceiling.
* [ ] Attention, FlashAttention block sizes, and XLA flag opportunities were considered early when the profile supports them.
* [ ] User approval is requested before quantizing beyond BF16 or adding a new custom Pallas kernel.
* [ ] Only one fix is tested at a time.
* [ ] The same scenario is re-profiled after each fix.
* [ ] Successful fixes are checkpointed when appropriate.
* [ ] Failed experiments are reverted without touching unrelated user changes.
* [ ] Diverging optimization paths are brought back to the user.
* [ ] The loop stops only when the model is squeezed enough or options are exhausted.

## Quick Routing Map
| Skill | Use It For |
| :--- | :--- |
| `skills/profiling/README.md` | XProf tools, hot ops, traces, graph viewer, trace windows, and profile reading. |
| `skills/roofline_analysis/README.md` | Module-level compute/HBM/communication ceilings and stop conditions. |
| `skills/jax/jax_efficiency.md` | JAX compilation, host sync, loops, vectorization, input pipeline, and NNX state. |
| `skills/torchax/torchax_efficiency.md` | PyTorch-on-JAX hygiene, JittableModule, torchax sharding, and fallback control. |
| `skills/performance/precision.md` | BF16/FP32 policy, Float64 promotion, dtype stability, and convert churn. |
| `skills/hardware/topology.md` | TPU physical topology, ICI/DCN, mesh creation, and axis ordering. |
| `skills/performance/sharding_and_layout.md` | DP/FSDP/CP/TP strategy, logical axis rules, layout mismatch, and collectives. |
| `skills/performance/attention.md` | Attention kernel selection, Ulysses/ring/local attention, and Pallas block tuning. |
| `skills/performance/xla_flags.md` | Compiler flag bundles, collective overlap, latency hiding, VMEM, and Sparse Core offload. |
| `skills/performance/fusion.md` | Fusion barriers, custom fusion decisions, and Pallas escalation. |
| `skills/kernels/SKILL.md` | Writing, updating, and optimizing distributed Pallas kernels on TPUs. |
| `skills/performance/quantization.md` | Low-precision quantization (FP8/INT8) on TPU after precision/fusion are settled. |
