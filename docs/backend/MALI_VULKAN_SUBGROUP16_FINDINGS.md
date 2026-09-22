# Mali (subgroup size 16) Vulkan op findings — MT6993 / Mali-G1-Ultra MC12

Measured on device, not inferred. Everything below has a command that reproduces it.

## Device under test

| item | value |
|---|---|
| device | Mali-G1-Ultra MC12 (MediaTek MT6993 / Dimensity-class, Android 16) |
| driver | v1.r54p1, Vulkan 1.3.305, UMA |
| backend flags | `uma: 1 \| fp16: 1 \| bf16: 0 \| fp4: 0 \| warp size: 16 \| shared memory: 32768 \| int dot: 1 \| matrix cores: KHR_coopmat` |
| host | Termux aarch64 (Android), llama.cpp master `fb34fc262c1b43f1832c7472429fb2247d650493` |

Build: `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON`, Release.
Run: `LD_LIBRARY_PATH=/system/lib64:$PREFIX/lib ./test-backend-ops <mode> -b Vulkan0`
(`-b` takes the *device* name, `Vulkan0`, not the backend name `Vulkan`; the wrong value silently skips every backend and exits 0.)

## Baseline (unmodified master)

`test-backend-ops -b Vulkan0`: **18880 OK / 53 FAIL / 3940 not-supported**, 136 op groups, exit 1.

| # | failure class | count | shape condition | pipeline used |
|---|---|---|---|---|
| 1 | `MUL_MAT` wrong results | 26 | `src0->ne[1] == 1` (m = 1), 21 quant types, ERR ≈ 0.2–1.0 | `matmul_quant_f16_f16acc_*` (coopmat1 MMQ); with `GGML_VK_DISABLE_COOPMAT=1` → `matmul_q4_0_q8_1_*` — **both fail** |
| 2 | `MUL_MAT` NaN / large ERR | 4 | `m=32,n=509,k=2112` (q8_0) → NaN; `m=32,n=509,k=2112`, `m=6,n=4096,k=5120` ERR ≈ 0.99 | same small-tile family |
| 3 | `MUL_MAT_ID` wrong results | 3 | `type_a=f16`, `m=32`, `k=16 / k=64` | `matmul_id_*` (f16) |
| 4 | `FLASH_ATTN_EXT` over tolerance | 20 | `hsk=hsv=72`, `nr23=[4,1]`, `kv=512`, `mask=0`, `nb∈{1,3}`, quantized K/V (q4_0/q4_1/q5_0/q5_1/q8_0), ERR 0.01–0.07 | `flash_attn_f32_f16` (same name with and without coopmat) |

Repro commands:

```bash
# class 1
./test-backend-ops -b Vulkan0 -o MUL_MAT -p "type_a=q4_0,type_b=f32,m=1,n=64,k=256"
# class 2
./test-backend-ops -b Vulkan0 -o MUL_MAT -p "type_a=q8_0,type_b=f32,m=32,n=509,k=2112"
# class 3
./test-backend-ops -b Vulkan0 -o MUL_MAT_ID -p "type_a=f16,n_mats=16,n_used=16,b=0,m=32,n=1024,k=16"
# class 4
./test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT -p "hsk=72,hsv=72,nh=4,nr23=.4,1.,kv=512,nb=1,mask=0,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=q4_0"
```

Which shader a case used: `GGML_VK_PIPELINE_STATS=<substring>` prints `pipeline stats for <pipeline name>`.

Non-findings (checked, ruled out):

* not a coopmat problem — `GGML_VK_DISABLE_COOPMAT=1` moves class 1 to the subgroup MMQ pipeline and it still fails;
* not an integer-dot or MMVQ problem — `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`, `GGML_VK_DISABLE_MMVQ=1` change nothing;
* not random test data — the suite seeds tensors with `std::random_device`, so `ERR` values differ per run, but pass/fail per shape is reproducible.

## Root cause of classes 1–2: small warptiles mis-tile at subgroup size 16

`mul_mm.comp` gives warp `warp_i` the corner at `warp_r = warp_i % (BM/WM)`, `warp_c = warp_i / (BM/WM)` and
stores at `ic * BN + warp_c * WN`. That is only sound when

```
(BM / WM) * (BN / WN) == BLOCK_SIZE / WARP
```

For the small warptiles `BM = BN = WN = 32`, so `WM` is determined by the pair `(BLOCK_SIZE, WARP)`, but
`ggml_vk_load_shaders` used a single shared constant:

```cpp
const uint32_t s_warptile_wm = device->subgroup_size == 8 ? 8 : 32;
```

At subgroup size 16 the small non-id tiles get `BLOCK_SIZE = max(16,32) = 32`, `WARP = max(16,8) = 16`, so
`BLOCK_SIZE/WARP = 2` while `(BM/WM)*(BN/WN) = 1`: `warp_c` reaches 1, the second warp reads `buf_b` out of
bounds and writes into the neighbouring workgroup's columns. Two workgroups own the same outputs with no
ordering between them — a race, hence ERR ≈ 1.0 and run-to-run variation.

The small pipeline is selected whenever `m <= 32 || n <= 32`, which is why this shows up as an `m=1` and
`n=509` failure rather than as a general breakage.

This is the same defect reported upstream as **ggml-org/llama.cpp issue #28637** ("Vulkan: small matmul
warptile mis-tiles at subgroup size 16, giving non-deterministic wrong results", opened 2026-09-09) on the
same GPU model, derived there by index replay only. This report supplies the missing on-hardware evidence.

### Fix tested here (WM from the identity, per warptile)

```cpp
auto const small_wm = [](uint32_t block_size, uint32_t warp) { return 32 * warp / block_size; };
const uint32_t s_warptile_wm       = small_wm(subgroup_size_32,         subgroup_size_8);
const uint32_t s_warptile_wm_id    = small_wm(mul_mat_subgroup_size_16, mul_mat_subgroup_size_16);
const uint32_t s_warptile_wm_qid   = small_wm(mul_mat_subgroup_size_32, mul_mat_subgroup_size_8);
const uint32_t s_warptile_wm_qid_k = small_wm(mul_mat_subgroup_size_32, mul_mat_subgroup_size_16);
```

with each of `s_warptile`, `s_warptile_mmq`, `_mmq_int`, `_mmq_int_k` using `s_warptile_wm`,
`s_warptile_id` using `_wm_id`, `s_warptile_mmqid`, `_mmqid_int` using `_wm_qid`, and
`s_warptile_mmqid_int_k` using `_wm_qid_k`.

Result on hardware (focused cases):

| case | before | after |
|---|---|---|
| `MUL_MAT(type_a=q4_0,type_b=f32,m=1,n=64,k=256)` | FAIL ERR 0.218 | **PASS** |
| same with q4_K / q8_0 / iq4_nl | FAIL | **PASS** |
| `MUL_MAT(type_a=q8_0,...,m=32,n=509,k=2112)` | NaN | **PASS** |
| `MUL_MAT(type_a=q4_0,...,m=16,n=1,k=4096)` (normal decode shape) | PASS | PASS (no regression) |
| `FLASH_ATTN_EXT(hsk=72,...,type_K=q4_0)` | FAIL ERR 0.052 | FAIL ERR 0.069 (unaffected, different shader) |

Full-suite rerun of the same build (`test-backend-ops -b Vulkan0`):

| | baseline | with the WM fix |
|---|---|---|
| cases OK | 18880 | **18912** |
| cases FAIL | **53** | **21** |
| log lines | 22945 | 22913 |
| exit code | 1 | 1 |

30 cases fixed, nothing regressed. What changed:

* all `MUL_MAT` failures with `m = 1` over quantized A (21 types) now pass, including the `bf16`-adjacent
  cases that were failing on the same shape;
* the odd-shape failures (`m=32,n=509,k=2112`, `m=6,n=4096,k=5120` and the `NaN` cases) now pass;
* the three `MUL_MAT_ID(type_a=f16, m=32, k=16/64)` cases now pass as well. Only the `_mmqid*` WM values
  changed for those tiles (`32 -> 16`), so the fix reached them, but which single tile is responsible has not
  been isolated by a per-tile build yet;
* one failure is left in the small-tile family and is **not** a tiling race: `MUL_MAT(type_a=bf16, m=1,
  n=64, k=256)` returns `NaN at index 33` (Vulkan `-nan`, CPU `-9.196352`). This device reports `bf16: 0`,
  so bf16 storage is emulated here; this needs its own investigation.

The 20 `FLASH_ATTN_EXT(hsk=72)` failures are unchanged by this fix and are tracked separately (P4 in the
validation plan). What is now known about them:

* exact boundary - `hsk=hsv=72, nh=4, nr23=[4,1], kv=512, mask=0, nb∈{1,3}`, K/V ∈ {q4_0, q4_1, q5_0, q5_1,
  q8_0}. All other combinations at `hsk=72` (f32/f16/bf16/iq4_nl, `nb∈{32,75}`, `mask=1`, `nr23=[1,1]`) pass,
  and q8_0 fails by the same margin as q4_0, so this is not a quantization-accuracy effect;
* the physical head size is 96, not 72: the test pads for quantized types
  (`tests/test-backend-ops.cpp:7781`, `hsk_padded = GGML_PAD(hsk, ggml_blck_size(type_K))`). The row stays
  block-aligned, so "72 is not a multiple of 32" is not the mechanism; that earlier guess is withdrawn;
* repro (1 minute): `./test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT -p "hsk=72"` → 20 of 1010 FAIL, ERR
  0.0175-0.0570 against a 0.0005 threshold;
* isolation: forcing `split_k = 1` in `ggml_vulkan.cpp` at the split-K selection (7955-7969) makes the same
  subset pass 1010/1010; forcing `split_k = 2` leaves 10 failures (only `nb=3`) and forcing `split_k = 64`
  leaves 20 with a larger mean error. The defect is therefore in the split-K path, and its magnitude scales
  with the number of splits;
* dispatch geometry (`GGML_VK_DEBUG_FA_SPLIT=1`, 1010 dispatches) narrows it further: at the failing shape the
  passing f16 case and the failing quantized case have identical `Br=4`, `split_k=8`, `split_kv=64` and
  `wg=(1,4,1)`; only `Bc` differs (64 vs 32). Non-GQA quantized cases with `Bc=32` and `split_k` up to 8 pass,
  so the remaining signature is GQA + `Bc=32` + `split_k>1`;
* exonerated: the K/V split window arithmetic (identical to the passing f16 case), the shared-memory K staging
  (`SHMEM_STAGING` is 0 off NVIDIA, so `kvsh` is not compiled in), and `use_dequant_kv` (that gate is false for
  all 1010 cases on this device, so forcing it away changed nothing - inconclusive, not negative);
* logs: `findings-splitk/tbo-fa-hsk72-{nosplitk,splitk2,bigsplitk}.log`, the dispatch log
  `fa-split-dbg.log`, plus the full-suite `tbo-vulkan-wmfix.log`.

### Also measured: the fix does not change output, and decode gets faster

Text equivalence against CPU (`compare_vk_cpu.sh`, generated text only) after the fix: `TEXT-IDENTICAL: yes`
for F16 and for F32, both producing `The capital of France is Paris.`

`llama-bench -ngl 99`, same two models, before → after:

| model | test | baseline | with the WM fix |
|---|---|---|---|
| qwen2 630M Q4_K_M | pp128 | 69.06 ± 0.50 | 70.91 ± 0.13 |
| qwen2 630M Q4_K_M | tg32 | 48.47 ± 5.35 | 54.23 ± 4.55 |
| qwen2 630M F16 | pp128 | 67.43 ± 1.63 | 67.12 ± 2.51 |
| qwen2 630M F16 | tg32 | 28.08 ± 2.18 | 37.95 ± 2.66 |

Decode moves the most (+12% to +35%); both runs are only two samples each, so the numbers are indicative
rather than a tuning claim. The point is that correcting the geometry does not cost performance here.

## Ops that are missing or gated, and which are worth adding here

Two questions get mixed up: "which ops never run on the GPU" and "which ops would a real model need". Measured
answers for this device:

**a) In a real model graph, nothing is missing.** `GGML_SCHED_DEBUG=2 ./llama-cli -v -ngl 99` on qwen2.5-0.5b F16
gives 486 nodes per graph evaluation with exactly **one** node off the GPU: `node #0 GET_ROWS(embd)`, the
token-embedding lookup, which llama.cpp keeps on the host for the input split by design. Everything else - 169
`MUL_MAT`, 120 `ADD`, 49 `RMS_NORM`, 49 `MUL`, 48 `ROPE`, 24 `FLASH_ATTN`, 24 `SWIGLU`, 2 `GET_ROWS` - runs on
Vulkan0. (Log: `sched-debug.log`.)

**b) The 12 ops that never go to the GPU are unused by this code base.** `DIAG_MASK_ZERO`, `IM2COL_BACK`,
`POOL_2D_BACK`, `FLASH_ATTN_BACK`, `WIN_PART`, `WIN_UNPART`, `GET_REL_POS`, `ADD_REL_POS`, `MAP_CUSTOM1/2/3`,
`CUSTOM`: **zero** references in `src/`, `examples/` or `tools/`. They exist for other ggml consumers (conv
training, Swin, T5 relative positions), not for llama-family inference. Adding them buys nothing here.

**c) The real gap is the training graph, not inference.** Every backward/optimizer op the training path uses
already has a Vulkan branch in `supports_op`: `OUT_PROD`, `CROSS_ENTROPY_LOSS(_BACK)`, `SOFT_MAX_BACK`,
`OPT_STEP_ADAMW`, `RMS_NORM_BACK`, `SILU_BACK`, `ROPE_BACK`, `ACC`, `GET_ROWS_BACK`, `REPEAT_BACK`. The only one
with **no** branch anywhere in `ggml-vulkan.cpp` is `FLASH_ATTN_BACK` (0 matches). Most training branches are
f32-only (`OUT_PROD`: contiguous f32/f32/f32; `ACC`: f32; `GET_ROWS_BACK`: f32; `CROSS_ENTROPY_LOSS`: f32), so
a mixed-precision training graph would still fall back to CPU. Consequence: `FLASH_ATTN_BACK` is the single
op-level item that would matter for training, but training does not reach the first step today (see the plan's
P6), and it is the heaviest of the remaining items.

**d) Gated shapes worth watching** (implemented, refused for specific inputs): `MUL_MAT` refuses non-contiguous
dim01 and `nr=[1,2]`; `FLASH_ATTN_EXT` refuses `q1_0/q2_0` KV; `CPY`/`SET_ROWS` use a type whitelist; `CONV_2D`
needs `cwhn=1`/specific dilations; `SSM_SCAN` (Mamba family) and `DSV4_HC_PRE` (DeepSeek-V4 hyper-connections)
each have a single refused case - those two extend real architecture coverage rather than correctness coverage.

## SME on this device: what the build actually produces

| step | measured result |
|---|---|
| hardware | all 8 cores report `sme sme2 smei8i32 smei16i32 smebi32i32 smef16f32 smef32f32 ... sve sve2 i8mm asimddp` (`/proc/cpuinfo`); two core types (`CPU part 0xd8b` x4, `0xd90` x4) |
| configure | `Performing Test HAVE_SME - Success`; `Adding CPU backend variant ggml-cpu: -march=armv9.2-a+sve2+sme` |
| baseline build (`GGML_CPU_ARM_ARCH` unset) | `quants.c.o` / `repack.cpp.o` / `vec.cpp.o` / `ops.cpp.o`: **0 SVE, 0 SME** instructions |
| SME build | same objects: **SVE 20 / 2 / 32 / 356**; 934 predicate-register uses in `ops.cpp.o`; ZA tile registers and `smstart`/`fmopa`: **0** |
| runtime + per-op correctness + bench | `sme-verify2.log`, `tbo-cpu-sme-mulmat.log` |

Clean CPU-only bench (idle device, `-ngl 0 -p 32 -n 8 -r 2 -t 4`, qwen2.5-0.5b F16):

| build | pp32 | tg8 |
|---|---|---|
| `-march=armv9.2-a+sve2+sme` | **209.47 t/s** | **21.63 t/s** |
| baseline, no `-march` (bare aarch64) | 16.85 t/s | 2.60 t/s |

That is a 12x prefill and 8x decode difference from the `-march` flag alone (single sample each in the first
pass; the r=2 rerun is in `sme-verify2.log`). It is the single highest-leverage switch for CPU fallback on this
device and costs nothing at runtime - but note it is SVE2 doing the work, not SME.

So `-march=armv9.2-a+sve2+sme` produces real SVE/SVE2 code but **no SME instructions at all** - nothing in
`ggml/src/ggml-cpu` uses SM/ZA. The only SME consumer in the tree is KleidiAI, gated behind
`-DGGML_CPU_KLEIDIAI=ON` (default OFF), which downloads `kleidiai-1.24.0-src.tar.gz` from ARM-software/kleidiai
via FetchContent (`ggml/src/ggml-cpu/CMakeLists.txt:586-613`) and provides forward matrix-multiplication
micro-kernels only. Enabling it is the only way to reach SME in inference here, and it cannot help training.

## Recent upstream Vulkan activity, filtered for ARM/Mali relevance

Checked against upstream `ggml-org/llama.cpp` (path `ggml/src/ggml-vulkan`, plus issues/PR search), because the
local clone is a depth-1 checkout and cannot answer this from `git log`:

* **no ARM/Mali-specific Vulkan commit is merged in the recent window.** The vendor-specific tile tuning that
  did land targets other vendors (Intel f16 B-type + warp tile tuning #27471, Intel Xe GEMM work, AMD RDNA
  workarounds, NVIDIA queuesubmit/argsort workarounds, PowerVR dmmv fallback #28341). The one merged commit
  that touches warp geometry generally is #27726 ("warptiles currently assume warp sizes <= 64, clamp to work
  around larger warps"), which was already present in the tested master;
* the defect this document is about is open as **#28637** (small matmul warptile mis-tiles at subgroup size 16),
  and it is the only open report filed against this exact GPU model;
* other open items worth tracking from the same sweep: **#28531** ("disable large matmul tile on Samsung GPUs
  with 32KB shared memory" — this device also reports `shared memory: 32768`, so the same policy question
  applies; the PR currently gates on `VK_VENDOR_ID_SAMSUNG` only), #29254 (F32 A matrix 2-aligned loads in
  `mul_mat_vec`), #29139 (hide internal symbols to prevent duplicate-dlopen state destruction — relevant to the
  Termux/Android dlopen case), #27703, #27952, #28507;
* Termux-specific toolchain reports: #28234 (Vulkan builds on Termux fail to optimise shaders) and the closed
  #13918 (`vulkan-shaders-gen` build for Termux);
* training: #18499 ("llama-finetune won't work even with 17M parameters") matches the abort reported in the
  validation plan;
* older Mali reports are closed/stale: #23241 (MUL_MAT wrong on Mali-G720), #23133 (Mali-G720 nonsense output),
  #23057 (descriptor set assert on ARM UMA), #26921 (Mali-G925 NaN logits), #23359 (Mali hang when the host app
  is backgrounded).

## Notes for the two open Mali PRs in this fork

* `vulkan: optimize ARM Mali subgroup warptiles` (upstream #27163, branch `vulkan-mali-optimization`) takes the
  other route: `subgroup_size_32 = device->subgroup_size` for ARM Mali. That satisfies the identity for the
  non-id small tiles, but leaves the three `_mmqid*` tiles on `mul_mat_subgroup_size_32 = max(16,32) = 32`
  with `WARP = 16`, i.e. still `2 != 1`. The identity-based fix covers those as well.
* The same branch no longer applies to current master: `vk_device_architecture` in
  `ggml/src/ggml-vulkan/ggml-vulkan-types.h` has no `ARM_MALI` member (only `OTHER`, `AMD_GCN`,
  `AMD_RDNA1..3`, `INTEL_XE1/XE2`, `NVIDIA_PRE_TURING/TURING`), so the guard fails to compile
  (`error: no member named 'ARM_MALI' in 'vk_device_architecture'`). It needs a rebase that either adds the
  architecture classification or drops it in favour of the identity fix.
* `vulkan: add runtime cache and Termux compatibility fixes` (upstream #27156) touches
  `ggml/src/ggml-vulkan/CMakeLists.txt` and `ggml/src/ggml-opt.cpp`; Termux build issues in the same area are
  also reported upstream as issue #28234.
