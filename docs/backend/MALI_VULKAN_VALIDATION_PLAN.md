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
   │  state: done                  │        │  still fails after the fix)        │
   │  1a + 1b shapes PASS          │        │  state: localized → split-K path   │
   │  full suite: 18912 OK / 21 FAIL│       │  split_k=1 → 0 FAIL (1010 cases)   │
   └──────────────┬────────────────┘        └───────────────┬────────────────────┘
                  │                                          │
                  ▼                                          │
   ┌───────────────────────────────┐                          │
   │ P3  _mmqid* warptiles         │                          │
   │  not covered by the shared    │                          │
   │  constant *or* by upstream    │                          │
   │  PR #27163 (mul_mat_          │                          │
   │  subgroup_size_32 stays 32)   │                          │
   │  state: 1c green; bf16 NaN    │                          │
   └──────────────┬────────────────┘                          │
                  │                                          │
                  ▼                                          ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P5  end-to-end inference regression                                          │
   │  Q4_K_M / F16 / F32 / F16+KV q8_0+FA: Vulkan text == CPU text (逐字一致)      │
   │  state: done — identical text after P2 too, decode not slower                 │
   └──────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P6  training / finetune path (Vulkan)                                        │
   │  llama-finetune -ngl 99 ; which ops stay on CPU ; sched offload behaviour     │
   │  state: answered — aborts before step 1, backend-independent                  │
   └──────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P7  SME path review (CPU side of the same question)                          │
   │  SME/SME2 present on this SoC but only reachable via KleidiAI (default OFF)   │
   │  state: answered — no -march in build, runtime shows NEON only               │
   └──────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P8  delivery                                                                 │
   │  a) upstream issue #28637: hardware confirmation + independent fix check      │
   │  b) our PR #27163: rebase (ARM_MALI enum gone, build fails) + _mmqid* tiles   │
   │  c) our PR #27156: rebase/revalidate                                          │
   │  state: docs pushed (this repo PR #4); upstream text is the owner's to write  │
   └──────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ P9  large-tile policy at 32KB shared memory                                  │
   │  upstream #28531 does this for Samsung; this device also reports 32768       │
   │  state: todo — perf only, correctness is P2                                  │
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

### P4 — FLASH_ATTN_EXT(hsk=72, quantized KV): localized to the split-K path

20 cases, ERR 0.0175–0.0570 against a 5e-4 tolerance, pipeline `flash_attn_f32_f16`, unaffected by the WM fix
and by `GGML_VK_DISABLE_COOPMAT`. `hsk=64/80/96/128/192/256/320/512/576` with quantized KV all pass.

**Correction after instrumenting the dispatch:** the physical head size is *not* 72. The test pads it for
quantized types (`tests/test-backend-ops.cpp:7781`, `hsk_padded = GGML_PAD(hsk, ggml_blck_size(type_K))`), so
these cases run with `HSK = HSV = 96` (three 32-element blocks per row) while carrying the name `hsk=72`. The
row width is block-aligned, so "hsk=72 is not a multiple of 32" is *not* the mechanism and was a wrong guess.

What the dispatch log (1010 cases, `GGML_VK_DEBUG_FA_SPLIT=1`) actually shows at the failing shape
(`nr23=[4,1], kv=512, mask=0`):

| case | Br | Bc | split_k | split_kv | wg | result |
|---|---|---|---|---|---|---|
| f16 / bf16 / f32 (`HSK=72`) | 4 | 64 | 8 | 64 | (1,4,1) | **OK** |
| quantized (`HSK=96`) | 4 | **32** | 8 | 64 | (1,4,1) | **FAIL** |
| quantized, non-GQA, `HSK=96` | 1-32 | 32/64 | 2..8 | 64..256 | - | **OK** |

So the geometry that fails is: GQA (`gqa_ratio=4`) + `Bc=32` + `split_k>1` + quantized K/V. The split window
arithmetic is identical to the passing f16 case (same `split_k=8`, `split_kv=64`, same workgroups), which
exonerates it. Two other candidate paths are also out: the shared-memory K staging (`SHMEM_STAGING` is 0 on
non-NVIDIA, so `kvsh` is not compiled in), and forcing `use_dequant_kv` had no effect because that gate is
false on this device for every one of the 1010 cases (`dequant_kv=0` throughout; the experiment is therefore
inconclusive, not negative).

The error is not a precision drift: q8_0 (nearly lossless) fails by the same amount as q4_0 — ERR between
0.0175 and 0.0570 against a 0.0005 threshold, and the values are type-independent across the five quantized
types. The full boundary (all four quadrants of the shape matrix) is in the findings document.

**Decisive isolation.** `split_k` was forced in the dispatch (`ggml_vulkan.cpp`, the block at 7955-7969) and the
whole `hsk=72` subset (1010 cases) was rerun each time:

| split_k | FAIL | which `nb` | ERR range | ERR mean |
|---|---|---|---|---|
| 1 (forced) | **0** | - | - | - |
| 2 (forced) | 10 | 3 | 0.0161 - 0.0390 | 0.0262 |
| natural (>1) | 20 | 1 and 3 | 0.0175 - 0.0569 | 0.0361 |
| 64 (forced) | 20 | 1 and 3 | 0.0181 - 0.0753 | 0.0431 |

So the failure lives in the split-K path, and its magnitude grows with the number of splits. With `split_k == 1`
the identical shapes and types pass, which is why this is invisible for normal batch sizes and only appears for
short sequences (few query rows make `split_k` large).

Candidate sites, in order (not yet pinned):

1. `Bc=32` under GQA + split. It is the only geometry difference between the failing quantized cases and the
   passing f16 cases at the same shape (`Br=4`, `split_k=8`, `split_kv=64`, `wg=(1,4,1)` on both sides), and
   non-GQA quantized cases with `Bc=32` and `split_k` up to 8 pass. Cleanest test: force `block_cols = 64` for
   the quantized GQA configuration (`get_fa_tuning_params_scalar`) and rerun the subset.
2. The per-split partial store/merge pair, specifically the GQA store branch (`flash_attn.comp:669-688`, which
   writes `o_offset` keyed by `(split, head, batch)` with the row inside) against the merge in
   `flash_attn_split_k_reduce.comp:34-35,103-105`. Checked by hand and index order agrees, but the hand check
   assumed `p.ne1` is the row count on both sides; a read-back of the partial buffers would settle it.
3. Nothing in the K/V window arithmetic (`flash_attn_base.glsl:181-182`): the f16 case at the same shape takes
   the same `split_k`/`split_kv`, so the windows are not the discriminator.

Next experiment (definitive if it works): read the split-K partial buffers back on the host for one failing
case and one passing case, recompute the merge manually, and see whether the partial `m`/`L`/`O` written by the
main shader are already wrong or whether the merge is.

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
