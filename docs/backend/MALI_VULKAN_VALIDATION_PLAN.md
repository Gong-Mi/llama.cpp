# Mali Vulkan validation plan (MT6993 / Mali-G1-Ultra MC12, subgroup size 16)

Plan graph for the work that follows `MALI_VULKAN_SUBGROUP16_FINDINGS.md`. Each node carries a state, the
evidence that sets it, and the command that closes it. States: `done`, `active`, `blocked`, `todo`.

## Graph

```text
                          ┌─────────────────────────────────────────┐
                          │ P0  baseline                             │
                          │ device + build + test harness            │
                          │ state: done                              │
                          └───────────────┬─────────────────────────┘
                                          │ produces: 18880 OK / 53 FAIL
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P1  shape-level triage                                                       │
   │  1a MUL_MAT m=1 (26)        1b MUL_MAT NaN/odd (4)                           │
   │  1c MUL_MAT_ID f16 small-k (3)   1d FLASH_ATTN_EXT hsk=72 (20)               │
   │  state: done — pipeline names + coopmat/int-dot/MMVQ exclusions recorded     │
   └──────────────┬──────────────────────────────────────┬────────────────────────┘
                  │                                      │
                  ▼                                      ▼
   ┌───────────────────────────────┐        ┌────────────────────────────────────┐
   │ P2  WM identity fix           │        │ P4  FLASH_ATTN_EXT hsk=72 + q KV   │
   │  (s_warptile_wm and friends)  │        │  independent of P2 (verified:      │
   │  state: done on hardware      │        │  still fails after the fix)        │
   │  1a + 1b shapes PASS          │        │  state: active — cause unknown     │
   │  full-suite rerun: see below  │        │  next: mm tile / K-quant dequant   │
   └──────────────┬────────────────┘        └───────────────┬────────────────────┘
                  │                                          │
                  ▼                                          │
   ┌───────────────────────────────┐                          │
   │ P3  _mmqid* warptiles         │                          │
   │  not covered by the shared    │                          │
   │  constant *or* by upstream    │                          │
   │  PR #27163 (mul_mat_          │                          │
   │  subgroup_size_32 stays 32)   │                          │
   │  state: todo — 1c still open  │                          │
   └──────────────┬────────────────┘                          │
                  │                                          │
                  ▼                                          ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P5  end-to-end inference regression                                          │
   │  Q4_K_M / F16 / F32 / F16+KV q8_0+FA: Vulkan text == CPU text (逐字一致)      │
   │  state: done on baseline; rerun after P2 before publishing                    │
   └──────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P6  training / finetune path (Vulkan)                                        │
   │  llama-finetune -ngl 99 ; which ops stay on CPU ; sched offload behaviour     │
   │  state: active                                                               │
   └──────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P7  SME path review (CPU side of the same question)                          │
   │  SME/SME2 present on this SoC but only reachable via KleidiAI (default OFF)   │
   │  state: active — build + capability evidence needed                          │
   └──────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P8  delivery                                                                 │
   │  a) upstream issue #28637: hardware confirmation + independent fix check      │
   │  b) our PR #27163: rebase (ARM_MALI enum gone) + extend to _mmqid* tiles      │
   │  c) our PR #27156: rebase/revalidate                                          │
   │  state: todo — upstream text must be written by the account owner             │
   └──────────────────────────────────────────────────────────────────────────────┘
```

## Node detail

### P0 — baseline (done)

`test-backend-ops -b Vulkan0`: 18880 OK / 53 FAIL / 3940 not-supported, exit 1, master `fb34fc2`.
`support` mode (supports_op only): 20331 cases / 111 ops, 16636 supported, 3695 refused
(100 ops fully supported; the refusals are shape/type boundaries such as non-contiguous dim01, `nr=[1,2]`,
`q1_0/q2_0` KV types, `mxfp4/nvfp4` copy targets).

### P1 — shape-level triage (done)

See the findings document. The important part is that class 1 is *not* tied to a backend feature: it survives
`GGML_VK_DISABLE_COOPMAT`, `_INTEGER_DOT_PRODUCT` and `_MMVQ`, which moves the question into tile geometry,
not feature selection.

### P2 — WM identity fix (done, acceptance met)

Four derived values replace one shared constant, so every small warptile gets a `WM` that satisfies
`(BM/WM)*(BN/WN) == BLOCK_SIZE/WARP` for its own `(BLOCK_SIZE, WARP)` pair.

Acceptance: class 1 and 2 cases pass, no previously passing shape regresses, and the total FAIL count drops.
Measured on hardware: 18880 → **18912 OK**, 53 → **21 FAIL**, 30 cases fixed, no regression.

### P3 — `_mmqid*` warptiles and the bf16 residue (partially done)

The three `s_warptile_mmqid*` tiles take `BLOCK_SIZE` from `mul_mat_subgroup_size_32` and `WARP` from
`mul_mat_subgroup_size_8/_16`, so at subgroup size 16 they still had `BLOCK_SIZE/WARP = 2` against
`BM/WM = 1`; the identity fix gives them `s_warptile_wm_qid(_k)` (`32 -> 16`). After the fix the three
`MUL_MAT_ID(type_a=f16, m=32, k=16/64)` cases pass. Open sub-items:

* isolate which of the four WM values accounts for the `MUL_MAT_ID` cases (one build per group);
* `MUL_MAT(type_a=bf16, m=1, n=64, k=256)` still returns `NaN at index 33` on this device (`bf16: 0`, so
  bf16 is emulated). This is a distinct defect, not the tiling race;
* the same identity check should be applied to the bf16 fallback tile (`s_warptile_wm_bf16`), which uses the
  same shared-constant pattern but cannot be exercised on this device.

Acceptance: the identity holds for every `subgroup_size` in {8, 16, 32, 64} by index replay, the three
`MUL_MAT_ID` cases stay green, and the bf16 `NaN` is either fixed or attributed.

### P4 — FLASH_ATTN_EXT `hsk=hsv=72` with quantized KV (active)

20 cases, ERR 0.01–0.07 against a 5e-4 tolerance, pipeline `flash_attn_f32_f16`, unaffected by the WM fix
and by `GGML_VK_DISABLE_COOPMAT`. Boundary that is known: the same shape with f32/f16/bf16/iq4_nl K/V passes,
`hsk=64/80/96/128/192/256/320/512/576` with quantized KV pass, and only `nr23=[4,1], kv=512, mask=0,
nb∈{1,3}` of the quantized `q*_0/q*_1/q8_0` cases fail.

Next steps: check the K-quant dequantisation path for `hsk=72` tile sizes in `flash_attn*` shaders (72 is not
a multiple of the usual 32/64 tile widths), and confirm whether the reference itself (CPU) uses the same
quantisation order.

### P5 — end-to-end regression (done, also after the fix)

`compare_vk_cpu.sh` compares generated text only (the `[ Prompt: ... | Generation: ... ]` line is not part of
the comparison). Baseline: Q4_K_M, F16, F32 and F16+KV q8_0+FA all produce identical text on Vulkan and CPU.
After the WM fix: still `TEXT-IDENTICAL: yes` (F16 and F32), and `llama-bench` shows decode at or above the
baseline (Q4_K_M tg32 48.5 → 54.2 t/s, F16 tg32 28.1 → 38.0 t/s, two samples each).

### P9 — 32KB shared-memory tile policy (todo, from the upstream sweep)

Upstream PR #28531 disables large matmul tiles for Samsung devices because at 32KB shared memory the large
tile quadruples accumulators per thread and collapses occupancy. This device reports the same 32768 bytes, so
the same experiment is worth running for ARM Mali: disable `mul_mat_l` / `mul_mat_id_l` for the Mali case and
measure `pp128+` with a larger `-ub`, then compare against the current large-tile selection. This is a
performance question only; P2 already fixed correctness.

Acceptance: prefill numbers with and without large tiles at `-ub 128/256`, sampled more than twice, with the
model, batch and quantization recorded.

### P6 — training / finetune path (answered: blocked, backend-independent)

`llama-finetune` on this device aborts before the first optimizer step, on **every** configuration tried:

| run | backend | result |
|---|---|---|
| `-c 128 -b 32 -ub 32 -opt sgd -lr 1e-5` | `-ngl 99` (Vulkan offload) | `GGML_ASSERT(cgraph->n_nodes < cgraph->size) failed` → SIGABRT, exit 134 |
| same | `-ngl 0` (CPU only) | same assert, same stack |
| `-c 64 -b 8 -ub 8` | `-ngl 0` | same assert, same stack |

Stack: `llama_context::opt_epoch_iter` → `ggml_opt_alloc` → `ggml_build_backward_expand` →
`ggml/src/ggml.c:7291`.

Mechanism, from the source: `llama_context::graph_max_nodes()` sizes the forward graph at
`max(1024, 8 * n_tensors)` = 2328 for this model (291 tensors, qwen2), while `ggml/src/ggml-opt.cpp:296`
creates the gradient graph with the *forward* graph's size (`ggml_new_graph_custom(ctx, src->size, true)`)
and then expands the backward pass into it. The backward expansion of a 24-layer model needs more nodes than
the forward budget, so the append helper aborts. Batch/context size is not the trigger (the `-b 8 -c 64` run
fails identically), and the backend is not the trigger (CPU-only fails identically).

So on this device: inference on Vulkan works, finetuning does not run at all, and the reason is a graph
capacity budget in the training path, not the Vulkan backend. The Vulkan forward pass itself completed
(the perf logger printed its table before the abort), which is also evidence that the training *forward* half
is functional on this backend.

Acceptance (met): the path is characterized with a reproducible abort, an exact file:line, and the
backend-independence shown by the `-ngl 0` runs.

### P7 — SME path review (answered: SME is not on any active path)

Evidence, all measured on this device:

* `/proc/cpuinfo` advertises `sme sme2 smei8i32 smei16i32 smebi32i32 smef16f32 smef32f32 smeb16f32` plus the
  full `sve/sve2` set;
* the build used here has **no `-march` at all**: configure prints `Checking for ARM features using flags:`
  with an empty list, because `GGML_NATIVE=OFF` with no `GGML_CPU_ARM_ARCH` / `GGML_CPU_ALL_VARIANTS` falls
  through the ARM branch in `ggml/src/ggml-cpu/CMakeLists.txt`;
* runtime confirmation: `system_info: ... CPU : NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | REPACK = 1 |` —
  no `DOTPROD`, no `I8MM`, no `SVE`, no `SME` line;
* in llama.cpp, SME is only reachable through KleidiAI (`GGML_USE_SME`, defined when `GGML_INTERNAL_SME` is
  set; its only consumer is `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`), and `GGML_CPU_KLEIDIAI` defaults to
  OFF. KleidiAI ships forward matmul micro-kernels only, so SME cannot participate in training;
* therefore "Vulkan + SME" is not a coupled path here: Vulkan does GPU work, the CPU backend does fallback
  work with plain NEON because the build has no ARM feature flags, and SME is dormant.

Acceptance (met for the capability matrix): inference = Vulkan (GPU) + NEON-only CPU fallback; training =
blocked before any backend matters (P6); SME = present in hardware, unused in this build, and unable to help
training even if enabled. A build with `-DGGML_CPU_ARM_ARCH=armv9.2-a+sve2+sme` or `-DGGML_CPU_KLEIDIAI=ON`
remains a separate measurement if the CPU side is worth accelerating.

### P8 — delivery (todo)

* Upstream issue #28637 asks for exactly the hardware confirmation this device can give, and proposes the
  same identity-based fix. The upstream text (issue comment, PR description) must be written by the account
  owner: llama.cpp's `AGENTS.md`/`CONTRIBUTING.md` forbid AI-written bug reports, PR descriptions and replies.
* Our upstream PR #27163 needs a rebase: `vk_device_architecture::ARM_MALI` does not exist in current master
  (`ggml/src/ggml-vulkan/ggml-vulkan-types.h`), so the branch no longer compiles, and its approach leaves the
  `_mmqid*` tiles unfixed (P3).
* Our upstream PR #27156 (`vulkan: add runtime cache and Termux compatibility fixes`) should be re-checked
  against current master, together with upstream issue #28234 (Termux shader optimisation failure).

## Open questions / not yet known

* Why exactly `hsk=72` and only `nb∈{1,3}` for the quantized KV flash-attention failures (P4).
* Which single WM value accounts for the three `MUL_MAT_ID` cases now passing (P3).
* Why `MUL_MAT(type_a=bf16, m=1, n=64, k=256)` still returns `NaN` on a device that emulates bf16 (P3).
* Whether the fixed batched shapes propagate to real MoE prefill on this device, which needs an MoE model to
  measure (`MUL_MAT_ID` is the prefill-heavy path for those architectures).
* Whether the training abort (P6) is specific to this model shape or general: a capacity fix in the training
  path (`ggml-opt.cpp` gradient graph size) is the next experiment, together with upstream issue #18499.
