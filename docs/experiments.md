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

`rejected/rotate_floor_analysis.cpp` isolates the cause:

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
`docs/performance.md`, "The FFN epilogue's row max".

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

## 6. Still untried

Splitting a group across 4 lanes so it is 64 elements per lane (4 zmm instead
of 16), with one cross-lane axis. A 512-bit cross-lane exchange needs 16
`select_from_group` calls, so the same verdict as the shared-memory version is
expected — but it is the only register-pressure reduction that has not actually
been built. Its upside is bounded: it cannot beat the 0.27 ms register-resident
floor, only approach the 0.12 ms streaming one.