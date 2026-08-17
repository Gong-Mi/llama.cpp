# Mali Vulkan warptile tuning: validation proposal

Status: draft proposal. This document does not change runtime warptile selection.

## Scope

Evaluate whether a Mali-G720 subgroup-16-specific matmul warptile can improve llama.cpp Vulkan performance without changing numerical results or affecting other vendors.

This is intentionally separate from the ARM Mali capability-reporting change. Capability detection is evidence collection; warptile selection is a performance change and requires a different acceptance bar.

## Current facts on the target device

Device:

- SoC: MediaTek Dimensity 9300 / MT6989
- GPU: Mali-G720-Immortalis MC12
- Vulkan driver: r44p1
- subgroup size: 16
- max compute shared memory: 32768 bytes
- Vulkan backend: working with 99 GPU layers

The current generic shader setup contains separate large, medium, and small warptile configurations. The current source also has an ARM vendor-specific medium-tile branch. The proposed experiment must therefore measure the exact affected pipeline rather than assume that every `subgroup_size_32` use is a Mali bottleneck.

## Candidate experiments

Each candidate must be a separate commit or branch:

1. Small-tile quantized matmul: alter only the Mali candidate warptile.
2. Non-quantized matmul: alter only the Mali candidate warptile.
3. Q8/int-dot matmul: alter only the Mali integer candidate.
4. Qi_K matmul: test separately because its register and shared-memory profile differs.

Do not combine these candidates in one patch. Do not change the generic subgroup-size constants as a proxy for warptile tuning.

## Correctness matrix

For every candidate, compare against the unmodified Vulkan build:

- Gemma 3 4B Q4_0
- Q8_0 and Q4_K workloads when available
- deterministic seed 42
- temperature 0
- prompt-only and prompt-plus-generation
- 99 GPU layers
- short context and a longer context
- at least two independent runs in both execution orders

Acceptance requires:

- process completes without Vulkan validation errors or device loss;
- no malformed, looping, or empty output where the baseline produces output;
- generated token IDs match the baseline for the deterministic test;
- no CPU fallback for the targeted matmul path;
- no change to non-Mali device selection logic.

## Performance matrix

Use the same model, build flags, linker workaround, device state, and command line:

```text
prompt: 128 and 512 tokens
generation: 128 and 256 tokens
repetitions: 3 minimum per cell
orders: baseline -> candidate and candidate -> baseline
metrics: prompt tok/s, generation tok/s, wall time, standard deviation
```

The first result is not sufficient. A candidate is only interesting if the improvement survives both execution orders and does not merely reflect DVFS or warm-up.

## Memory and shader checks

For each candidate:

- verify computed shared-memory use is below 32768 bytes;
- capture the selected pipeline/warptile when debug logging is enabled;
- verify shader generation and pipeline creation complete on the target driver;
- record model, context, compute-buffer, and KV-cache memory separately;
- do not use the current `unaccounted` memory field as a precise allocation measurement until its accounting is fixed.

## Current evidence boundary

The existing capability-report PR proves ARM Mali identification and runtime capability reporting. It does not prove a warptile performance improvement.

A previous experiment that directly changed the generic subgroup-32 specialization did not produce a reproducible performance benefit. That experiment is not evidence against a properly isolated warptile candidate; it is evidence that subgroup-size replacement and warptile tuning must not be conflated.

This draft remains documentation-only until one candidate passes the correctness and ordered benchmark matrix above.

## Evidence collected after opening this draft

The current upstream source already contains an ARM vendor-specific medium-tile selection in `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (the branch beginning at `device->vendor_id == VK_VENDOR_ID_ARM && device->subgroup_size >= 16`). A new patch must not duplicate that branch or claim it as new work.

On the target device, the capability-report build produced this baseline with the existing runtime selection:

```text
model: Gemma 3 4B Q4_0
GPU layers: 99
prompt: 128 tokens, 3 repetitions
generation: 128 tokens, 3 repetitions
prompt: 60.0363 tok/s ± 0.1581
generation: 10.3982 tok/s ± 0.3375
```

A direct experiment that replaced the generic subgroup-32 specialization with subgroup 16 was also run, but it is not a valid warptile patch and did not provide reproducible evidence. Ordered runs were affected by device thermal/DVFS state. It is therefore excluded from the runtime patch set.

The next runtime candidate, if pursued, must target one exact existing pipeline (for example small quantized MMQ) and change only its warptile values. The candidate must first demonstrate that it is not already covered by the existing ARM medium-tile branch.
