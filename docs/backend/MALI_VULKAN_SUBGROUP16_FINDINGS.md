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
validation plan).

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
