# Experiment log

Chronological record of what was tried, what worked, and what did not. Kept so
that a future attempt does not repeat a dead end. Numbers are at M=4096,
K=N=3072 unless noted.

Legend: **WIN** kept in the library · **DEAD** measured slower or neutral ·
**WRONG** faulty result that looked promising.

---

## 1. INT8 GEMM — 0.71 → 43 TOPS

The first ESIMD GEMM was 36x slower than `torch._int_mm`. Three plain bugs:

| fix | result |
|---|---|
| `for(i<BN) B[i]=bblk[i]` — 512 **scalar byte loads** per k-chunk → `memcpy` | 0.71 → 6.1 TOPS |
| `acc[j]=acc[j]+t[j]` (16 scalar adds) → one `vec` add, `memcpy` store | 6.1 → 8.6 |
| register-file tiling + `ROWS` sweep | 8.6 → **43.4** |

`ROWS` sweep: 2 → 12.1 (B-load bound), 4 → 23.7, **8 → 43.4**, 12 → 28 (spill),
16 → 6.0 (heavy spill). Rows=8 is the register-file optimum: 8 accumulators +
8 zmm of B + A + one DPAS result ≈ 18 of 32 zmm.

**DEAD — `kernels/int8_gemm_v4.cpp`: DPAS accumulator + RepeatCount.** ESIMD
also exposes `dpas<8,R,T,CT,BT,AT,...,N,BN,AN>(C,B,A) == C + A×B` with
`AN == R*32` bytes for R rows. This cuts instructions per unit of work by 2.5x
(13 instead of 32 per 8 units) by folding the accumulate into the instruction
and amortising the B load over R rows. CPU-verified correct at R=1,2,4,8.
**Measured 28 TOPS at R=8 vs 43 for v3.** The 8-zmm accumulator operand costs
more than the instructions saved.

**Where v3 actually stands:** L2-bandwidth bound at ~4 TB/s (see
`docs/performance.md`). Instruction count is not the limit, which is why the
2.5x instruction reduction in v4 bought nothing.

---

## 2. rotate + quantize — 0.765 → 0.52 ms

### Wins

| change | result |
|---|---|
| explicit accumulation chain in `inreg()` instead of one 4-product expression | 0.63 → 0.55 |
| load + transform one axis-2 group at a time instead of all 16 then all | 0.63 → 0.50 |
| compile-time `W` indices (template on groups-per-sub-group) | avoids a ~35% scratch-memory demotion |
| `gid >> 4` / `local_id & 15` instead of division | minor |
| **BF16 kernel: flat store loop** instead of interleaved butterfly+store | **1.44x** |

The BF16 store-pattern finding matters beyond its size: the interleaved version
was both slower **and wrong** (3.3e-02 error vs float64, versus 3.1e-09 for
the flat loop). Register-resident arrays plus an interleaved butterfly are a
shape this compiler gets wrong.

### The structural finding

`debug/native/rejected/rotate_floor_analysis.cpp` isolates the cause:

```
same copy loop, 16 registers held live   0.402 ms
same copy loop, load/store interleaved    0.128 ms
```

3.1x from register residency alone. The rotation's row abs-max means all 256
values must stay live until the scale is known, so all 16 loads complete before
the first store. The sub-group reduce itself costs 0.000 ms — it is the
serialisation, not the reduce.

### Dead ends

| idea | measurement | note |
|---|---|---|
| fp16 butterfly (8 zmm instead of 16) | **1.14 ms** | no native 512-bit fp16 vector ops on this path; register pressure was never the constraint |
| shared-memory staging of the strided axis (16 → 4 zmm peak) | **1.33 ms** | barriers cost more than the pressure |
| 3-pass: rotate+stage bf16 / reduce / quantize | **0.75 ms** | predicted 0.27. Pass 1 alone is 0.73 ms — the 16 registers are still live |
| forced full unrolling of the group loop | **1.27 ms** | spills |
| work-group size 16/32/64/128/256 | 0.60/0.60/0.57/0.57/0.59 | flat, not a lever |
| 64-byte stores instead of 16-byte | no change |
| 64-byte loads instead of 32-byte | no change (0.995x paired A/B, bit-identical) | first measured as a 10% regression by separate runs -- thermally confounded, retracted |
| `vec<int32_t>` pack instead of scalar float packs | no change | |
| AOT `spir64_gen12` | unavailable | this compiler accepts only generic `spir64` |
| idle lanes (4/16 idle at K=3072) | not a cost | K=1024 with 4/16 busy is the *fastest* per element — cache size |
| 2 / 4 rows per work-group | **0.76 / 0.42 ms** | 1.6-3x loss |
| group split across 4 lanes, one cross-lane axis | **not buildable at K=3072** | the exchange is cheap (84 ns per 512-bit move, 36 of them per lane per row = ~3 us against a 250 us prize) but the row's abs-max would have to cross 3 sub-groups; see `docs/performance.md`, "What is left", and `debug/native/probe_cross_lane.cpp` |

---

## 3. dequant epilogue — no headroom

| idea | measurement |
|---|---|
| scalar `r[t]=BF(v[t])` (production) | 0.135 ms |
| rounding-bit trick + scalar pack | 0.256 ms |
| rounding-bit trick + swizzle pack | 0.239 ms |
| **bandwidth floor for 75 MB** | **0.12 ms** |

At **88% of the floor**. The scalar loop is already vectorised; the "obvious"
bit trick is 2x worse. No headroom.

The same kernel's *row max* is a different story: the barrier + serial
local-memory loop that gates every store costs 24 % of the FFN epilogue, and one
`sycl::reduce_over_group` gets 16 % of the kernel back (bit-identical output,
interleaved) -- not shipped because the construct once killed the host process.
`docs/performance.md`, "The FFN epilogue's row max", and the report to the
vendor in `docs/report-reduce-over-group.md`.

---

## 4. Pipeline-level

| idea | measurement |
|---|---|
| N-block the GEMM so `acc` stays in the 18 MB L2 | blocked `_int_mm` 0.99 ms vs 0.50 monolithic, vs 0.13 saved on the epilogue |
| `_int_mm` with `[N,K].t()` vs `[K,N]` contiguous | within noise |
| `_weight_int8pack_mm` | requires a **float** A — not an int8xint8 path |
| `torch.empty` per call | 0.004 ms regardless of size |
| fusing the epilogue into our own GEMM | GEMM would need 151 TOPS to break even; it has 43 and is L2-bound |

---

## 5. Two measurement bugs worth remembering

**The ablation that stored twice.** The first version of
`probe_rotate_ablation.cpp` wrote the output in two places, which inflated the
store attribution from 0.131 to 0.215 ms and sent a round of optimisation at
the int8 pack. The corrected numbers are in `docs/performance.md`; the probe's
header warns about it.

**ctypes truncating pointers.** `ctypes` defaults unannotated integer
arguments to a 32-bit C `int`. Calling a `void*`-taking function without
setting `argtypes` truncates a 64-bit `tensor.data_ptr()`, the kernel writes
through garbage, and the symptom is `UR_RESULT_ERROR_DEVICE_LOST` — not an
argument error. I misdiagnosed this twice as a kernel resource problem before
bisecting it. Always go through `debug/common.load_lib()`.

---

## 5b. The weight dequantizer as a matrix product: Triton's `tl.dot` is 20-100x below oneDNN here, `torch.mm` is not

The idea: the dequantizer has no row abs-max, so the group rotation need not be a butterfly in one lane's 16 registers; as a
256x256 product `a @ R` (R = the library's rotation applied to the identity) it goes to the matrix unit. The first version did it
inside one Triton kernel and was **30-400 times slower** than the SYCL kernel (22-170 ms against 0.05-0.34 ms; most launch
configurations did not build: "total scratch space exceeds HW supported limit").

`debug/probe_triton_dot.py` (1280x5120, a [25600, 256] x [256, 256] product, 3.36 GFLOP) found it is `tl.dot` itself, not the
dequantization around it:

| | time | TFLOPS |
|---|---|---|
| `torch.mm` (oneDNN) | 0.062-0.083 ms | 41-54 |
| Triton, plain pointers, bf16 A, tile 64x64, 8 warps | 6.1 ms | 0.5 |
| the same, tile 128x64 or 4 warps | 11.8-12.4 ms | 0.3 |
| Triton, block pointers, tile 64x64, 8 warps | 1.57 ms | 2.1 |
| + int8 operand widened in registers | 7.7 ms | 0.4 |
| + row scale and bf16 rounding | 7.4 ms | 0.5 |
| the first version (whole 256-wide operand, one product) | 51 ms | 0.1 |

The int8 widening and the scale cost nothing measurable; the Triton matrix product is two orders of magnitude below the
vendor library on this card with these tile sizes (small; larger tiles and more warps were not tried). **Do not build a kernel
on `tl.dot` here without a tuned configuration that reaches tens of TFLOPS first.**

What works: a Triton elementwise pass (dequantize and round to bf16, 0.03 ms at 1280x5120, 225 GB/s) followed by `torch.mm`
(0.055 ms): `xpu_kernels.triton.dequant`, bit-identical to the SYCL kernel on every value tested (0 of 13 M), 1.3-5.1x faster in
device time at SDXL's shapes (`debug/bench_dequant_triton.py`).

## 6. Still untried

Nothing, for the rotate: the last open shape (splitting a group across 4
lanes) was built and measured, and the answer was that the obstacle is not the
exchange but the *reduction domain* — splitting the groups makes the row's
scale cross sub-groups, which costs a barrier (1.33 ms) or a second pass
(0.75 ms) against the 0.402 ms the rotate costs now. See
`docs/performance.md`, "What is left: the rotate's floor", and
`debug/native/probe_cross_lane.cpp`. The route that would work — a scale per
block instead of per row — changes the numbers, so it is a design decision
rather than an optimisation.