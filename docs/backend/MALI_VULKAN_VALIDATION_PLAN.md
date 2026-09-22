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

### P2 — WM identity fix (done on hardware, full-suite rerun pending)

Four derived values replace one shared constant, so every small warptile gets a `WM` that satisfies
`(BM/WM)*(BN/WN) == BLOCK_SIZE/WARP` for its own `(BLOCK_SIZE, WARP)` pair. Hardware results for the focused
cases are in the findings document; the full-suite rerun (`tbo-vulkan-wmfix.log`) gives the new FAIL total.

Acceptance: class 1 and 2 cases pass, no previously passing shape regresses, and the total FAIL count drops
by 29–30 (the classes 1+2 population).

### P3 — `_mmqid*` warptiles (todo)

The three `s_warptile_mmqid*` tiles take `BLOCK_SIZE` from `mul_mat_subgroup_size_32` and `WARP` from
`mul_mat_subgroup_size_8/_16`, so at subgroup size 16 they still have `BLOCK_SIZE/WARP = 2` against
`BM/WM = 1`. The identity fix above assigns them `s_warptile_wm_qid(_k)`, which is why they are expected to
improve; the remaining `MUL_MAT_ID(type_a=f16, m=32, k=16/64)` failures use the *f16* id path and are a
separate question that this node owns.

Acceptance: the three `_mmqid*` tiles satisfy the identity for every `subgroup_size` in {8, 16, 32, 64}, shown
by index replay, plus hardware reruns of the class 3 shapes.

### P4 — FLASH_ATTN_EXT `hsk=hsv=72` with quantized KV (active)

20 cases, ERR 0.01–0.07 against a 5e-4 tolerance, pipeline `flash_attn_f32_f16`, unaffected by the WM fix
and by `GGML_VK_DISABLE_COOPMAT`. Boundary that is known: the same shape with f32/f16/bf16/iq4_nl K/V passes,
`hsk=64/80/96/128/192/256/320/512/576` with quantized KV pass, and only `nr23=[4,1], kv=512, mask=0,
nb∈{1,3}` of the quantized `q*_0/q*_1/q8_0` cases fail.

Next steps: check the K-quant dequantisation path for `hsk=72` tile sizes in `flash_attn*` shaders (72 is not
a multiple of the usual 32/64 tile widths), and confirm whether the reference itself (CPU) uses the same
quantisation order.

### P5 — end-to-end regression (done on baseline)

`compare_vk_cpu.sh` compares generated text only (the `[ Prompt: ... | Generation: ... ]` line is not part of
the comparison). Baseline: Q4_K_M, F16, F32 and F16+KV q8_0+FA all produce identical text on Vulkan and CPU.
Rerun after P2 to make sure the tile change does not alter decode output.

### P6 — training / finetune path (active)

`llama-finetune` drives `llama_opt_init` → `ggml_opt_*` over the *context's* scheduler, so GPU backends are
eligible. Known constraints found in-tree:

* FA is switched off for training because `FLASH_ATTN_EXT` has no backward pass;
* `OUT_PROD` in the Vulkan backend is f32-only, which is why `finetune.cpp` forces f32 K/V caches;
* Vulkan has no `IM2COL_BACK`, `POOL_2D_BACK`, `FLASH_ATTN_BACK`, `DIAG_MASK_ZERO`, `WIN_PART/UNPART`,
  `GET_REL_POS/ADD_REL_POS`, `MAP_CUSTOM*`, `CUSTOM` — those nodes fall back to CPU through the scheduler;
* `CROSS_ENTROPY_LOSS(_BACK)`, `SOFT_MAX_BACK`, `RMS_NORM_BACK`, `SILU_BACK`, `ROPE_BACK`, `ACC`,
  `OPT_STEP_ADAMW/SGD`, `GET_ROWS_BACK`, `REPEAT_BACK` all have Vulkan f32 pipelines.

Acceptance: a run with `-ngl 99` either completes an epoch and writes the finetuned model, or fails with a
specific op/backend attribution. Both outcomes get recorded; the point is to know which it is.

### P7 — SME path review (active)

Evidence collected so far:

* this SoC advertises `sme sme2 smei8i32 smei16i32 smebi32i32 smef16f32 smef32f32 smeb16f32` plus the full
  `sve/sve2` set in `/proc/cpuinfo`;
* in llama.cpp, SME is reachable only through the KleidiAI path: `GGML_USE_SME` is defined by
  `ggml/src/ggml-cpu/CMakeLists.txt` when `GGML_INTERNAL_SME` is set, and its only consumer is
  `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp` (`if (!af.has_sme) { return 0; }`), while
  `ggml/src/ggml-cpu/ggml-cpu.c` only reports capability (`ggml_cpu_has_sme/sme2`);
* `GGML_CPU_KLEIDIAI` defaults to OFF, so a default build (including this one) neither compiles nor runs SME
  kernels; KleidiAI provides forward matmul micro-kernels only, so SME cannot contribute to training;
* therefore "Vulkan for training/inference + SME on the CPU side" is currently two independent paths, and
  the Vulkan backend does not interact with SME at all.

Acceptance: a capability matrix stating, per path (forward inference, training backward, finetune), which
backend does the work and whether SME is enabled; plus a measured build with `GGML_CPU_KLEIDIAI=ON` for the
inference path if it is worth enabling on this SoC.

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
* Whether the f16 `MUL_MAT_ID` failures share the tile geometry cause or are an independent shader bug (P3).
* Whether the Vulkan training path is viable at all on this device, or whether the scheduler ends up pushing
  every backward node to CPU (P6) — no upstream record either way yet; upstream issue #18499 reports
  `llama-finetune` failing in general.
