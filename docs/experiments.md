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
on `tl.dot` here without a tuned configuration that reaches tens of TFLOPS first.** That configuration exists and it is not a
tile size: the matrix units have to be *visible to the compiler*, which on this stack takes an environment variable (5c). After
that `tl.dot` reaches ~52 TFLOPS.

A Triton elementwise pass followed by `torch.mm` did beat the old SYCL kernel 1.3-5.1x (bit-identical), but it was a detour: the
old kernel was slow because a lane held a whole 256-element group (256 live floats, spilled) and loaded and stored scalars
strided across lanes. One sub-group per group with a local-memory transpose (`debug/native/probe_dequant_weight.cpp` modes 8, 9)
is bit-identical and runs 0.034 ms at 1280x5120 and 0.109 ms at 10240x1280 in the library, 0.33-0.82x the time of the two-pass
form's 0.103 and 0.240 ms; the two-pass module was removed. The probe's "floor" mode 6 was itself slow at 10240x1280 only because
the runtime picked the work-group size: with an explicit 64 (mode 10) it takes 0.086 ms against 0.153 ms.

## 5c. The Triton attention on the B580: the matrix units were hidden behind an OpenCL platform, then 52 TFLOPS

`xpu_kernels.triton.attention` (forward and a deterministic two-kernel backward) was the test case 5b left open. It compiled,
agreed with torch and captured, but was 20-100x slower than oneDNN, as 5b predicted. The cause was not the kernel: Triton's Intel
backend asks every OpenCL platform for the device extensions (`triton/backends/intel/extension_utils.c`), and the Mesa rusticl
platform answers for the same B580 without `cl_intel_subgroup_matrix_multiply_accumulate`. The target then says the card has no
XMX and `tl.dot` is lowered to FMA. Triton's own environment knob overrides the query; importing `xpu_kernels` now sets the
two gate-safe extensions when the variable is unset (`os.environ.setdefault`, before any Triton import; a value already in the
environment wins). The three-extension list below is the measured opt-in:

```
TRITON_INTEL_DEVICE_EXTENSIONS="cl_intel_subgroup_matrix_multiply_accumulate cl_intel_subgroup_2d_block_io cl_intel_bfloat16_conversions"
```

Same kernel, same shape (forward, 1024x1024, 10 heads, batch 4, device time, min of 3):

| target | time | TFLOPS |
|---|---|---|
| no XMX (FMA) | 38.631 ms | 0.28 |
| XMX | 0.206 ms | 52 |

187x. A direct `attention()` call still runs either way, but `usable()` now refuses when the compiler target lacks the extension:
the FMA path is far slower than the math attention this exists to replace, and the module logs one warning naming the variable.
torch is not a second opinion here: `torch.xpu.get_device_capability` reported the extension `False` in some processes and `True` in
others on this machine, while Triton's target stayed consistent, so only Triton's answer is used.

**Descriptors.** `tl.make_tensor_descriptor` built inside the kernel (with `triton.set_allocator`) is faster than pointer
arithmetic and than a host `TensorDescriptor`, forward and backward, and it still captures. There is one descriptor per head, so
rows past `Lq` clip at the head's end instead of running into the next head, which is what the masked pointer stores had to
handle. 1024x1024, 10 heads, batch 4, one config each:

| load path | forward | dQ | dK/dV |
|---|---|---|---|
| pointers | 0.572 | 0.820 | 1.206 |
| device descriptors | 0.216 | 0.343 | 0.609 |

Re-tuning per kernel then beat those numbers: the backward's two kernels do different work and sharing the forward's tiles is
bad. dK/dV with the forward's (128, 64) spilled: 15.8 ms, 13x the tuned (16, 64).

**Final configuration** for head dimension 64 (the only one measured, SDXL's): forward `(128, 32, 8 warps, 2 stages)`, dQ
`(128, 16, 8, 2)`, dK/dV `(16, 64, 8, 2)`. The card's clock state moves the forward's ranking; interleaved with n=100 over ten
rounds it was (128,32) 0.211-0.288 ms, (64,32) 0.218-0.436, (64,64) 0.340-0.369, so (128,32) is baked in (best sustained, within
5 % of the best fast). dQ (128,16) and dK/dV (16,64) won in both states.

**End to end** (`debug/verify_attention_capture.py`, batch 4, forward+backward through autograd, min of 3 interleaved rounds, all
three extensions set):

| shape | this | flash | math | this/flash | math/this |
|---|---|---|---|---|---|
| self 1024 (10 heads) | 1.576 | 1.556 | 6.824 | 1.01x | 4.33x |
| cross 77 (10 heads) | 0.328 | 0.404 | 1.349 | 0.81x | 4.11x |
| self 256 (20 heads) | 0.307 | 0.208 | 1.213 | 1.48x | 3.95x |
| cross 77 (20 heads) | 0.155 | 0.110 | 0.500 | 1.42x | 3.22x |

The target was at least 2x the math backend and within ~1.3x of flash: met at self-1024 and both 10-head cross shapes; the
20-head 256-token shapes are 1.42-1.48x flash but 3.2-4.0x math. Capture and replay on new data are bit-identical for out, dq,
dk and dv; the errors against a float32 chain equal torch's own bf16 attention to the last digit (2.1-2.4e-3 on all four).

**The third extension has a price.** `cl_intel_bfloat16_conversions` carries most of the last 1.9x (forward, batch 4
1024x1024: 0.391 ms with `..._matrix_multiply_accumulate ..._2d_block_io`, 0.206 ms with the conversion extension as well), but
with it set the rope's bit-exact gate finds 33 of 1,572,864 values different in `rms_rope` and 32 in `rms_rope_dequant`,
deterministically, and falls back to the older fused kernel. That is a bf16-rounding difference between the hardware conversion
and the emulated one -- the gate doing its job, not a wrong result -- so the full list is not free. With only the two extensions
the same verify run reads:

| shape | this | flash | math | this/flash | math/this |
|---|---|---|---|---|---|
| self 1024 (10 heads) | 1.924 | 1.403 | 6.780 | 1.37x | 3.52x |
| cross 77 (10 heads) | 0.428 | 0.255 | 1.040 | 1.68x | 2.43x |
| self 256 (20 heads) | 0.430 | 0.227 | 1.208 | 1.89x | 2.81x |
| cross 77 (20 heads) | 0.225 | 0.104 | 0.511 | 2.17x | 2.27x |

The at-least-2x-math target still holds everywhere; the within-1.3x-of-flash one only at self-1024. Which list to standardise is
the owner's call: all three favour the attention, the two-extension list keeps every gate in the library open.

The time readings wander by up to ~1.8x between a cold process and a sustained one. The ratios above come from interleaved
rounds in one process; the first version of part 4 measured the three paths one after another and printed 1.29x/3.24x where the
interleaved run printed 1.01x/4.33x, which is why it now interleaves.


## 5d. The attention in a whole captured training step: 1.68x at 512, 1.07x at 1024

The trainer's question after 5c was what the kernel does to a *step*. Measured with `b580-diffusion-training` at 577c96c
(`trainer.graph_attention=triton`), 200 steps, SDXL LoRA bf16, `--rank 16 --batch-area 1.5 --param buckets.max_shapes=4`, same
checkpoint/dataset/seed, torch 2.12.1+xpu, triton-xpu 3.7.1. The XMX setup used for these runs was
`TRITON_INTEL_DEVICE_EXTENSIONS="cl_intel_subgroup_matrix_multiply_accumulate cl_intel_subgroup_2d_block_io"`; importing
`xpu_kernels` now sets that same default itself, so the export is no longer needed (an explicit value wins). Without the units
`usable()` is False, with them every run logged `graph attention: the fused kernel`. Steady state is steps 21-200, median [mean];
the mean includes the one-off first-sighting steps (3.6-4.3 s).

| run | ms/step | peak device | loss@50 / @100 / @200 |
|---|---|---|---|
| 512/4 eager | 996 [1028] | 8357 MB | 0.1573 / 0.1021 / 0.1028 |
| 512/4 graph + math | 923 [966] | 10025 MB | 0.1573 / 0.1021 / 0.1028 |
| 512/4 graph + math, repeat | 968 [1004] | 10005 MB | 0.1573 / 0.1021 / 0.1028 |
| 512/4 graph + triton | 593 [688] | 9410 MB | 0.1573 / 0.1020 / 0.1027 |
| 512/4 graph + triton, repeat | 618 [716] | 9275 MB | 0.1572 / 0.1021 / 0.1027 |
| 1024/1 eager | 1036 [1093] | 8554 MB | 0.2063 / 0.0914 / 0.1304 |
| 1024/1 graph + math | 1014 [1083] | 11775 MB | 0.2062 / 0.0914 / 0.1302 |
| 1024/1 graph + math, repeat | 1014 [1078] | 11721 MB | 0.2063 / 0.0914 / 0.1306 |
| 1024/1 graph + triton | 963 [1040] | 10088 MB | 0.2061 / 0.0914 / 0.1304 |
| 1024/1 graph + triton, repeat | 972 [1055] | 10144 MB | 0.2062 / 0.0913 / 0.1303 |

`triton` is faster than eager at both sizes: 1.68x at 512/4 (1.49x by mean) and 1.07x at 1024/1 (1.05x by mean). The per-shape
pairs the trainer measures in-run are 27-42 % at 512 and 2-9 % at 1024:

| 512/4 shape | replay | eager | 1024/1 shape | replay | eager |
|---|---|---|---|---|---|
| 3x100x64 | 558 / 554 | 967 / 917 | 1x136x184 | 883 / 891 | 970 / 969 |
| 3x64x128 | 665 / 674 | 950 / 945 | 1x204x128 | 922 / 908 | 977 / 975 |
| 3x68x92 | 546 / 542 | 933 / 926 | 1x252x128 | 1111 / 1123 | 1187 / 1144 (run 2 drops it) |
| 3x128x64 | 685 / 695 | 944 / 949 | 1x128x256 | 1116 / 1125 | 1155 / 1170 (both drop it) |

The math graph does not pay and the trainer drops it: at 512 it replayed 890 against 906 ms eager (then the memory cap refused
the other shapes); at 1024 its first capture ran out of memory (1.46 GiB request, 11.8 GB peak) and nothing replays. The triton
run captures all four bucket shapes; at 512 the two rare short-batch shapes are outside `shape_counts()` and stay eager at
~930 ms, and at 1024 the shapes replaying under 5 % faster are dropped (three kept in one run, two in the other). The graph adds
memory outside the step's own peak: the allocator holds 7338 MB at the end of the 512 run against 6658 eager, and 8244 against
6888 at 1024, while the per-step footprint is 6479 against 6472 MB at 512 and 6782 against 6778 MB at 1024.

Loss follows `math` inside the repeated-arm noise: at 512 the two math runs read 0.1028 at step 200 and the two triton runs
0.1027; at 1024 math reads 0.1302/0.1306 and triton 0.1304/0.1303. The three-extension list gives 591 [680] ms against
593/618 [688/716] and the same loss, so it buys nothing at the step level; the two-extension default stays.

`debug/verify_attention_capture.py` now captures both cross-77 and self-1024: replay of `out`, `dq`, `dk`, `dv` is identical on
both, and the errors against float32 equal torch's own bf16 attention (2.2e-3 out, 2.4e-3 dq on self-1024; forward and backward
timings unchanged from 5c).

Profiler of the 512/4 triton step (`python -m run ... --perf DIR`, 70 warm-up steps, 3 timed, 3 eager-profiled): replay 608 ms,
device busy 564 ms (93 % utilisation), 22,889 kernel launches and 195,939 operators per step, 286 us host time per launch. The
fused attention is **71.8 ms, 12.7 % of device time**, in 700 launches (`_fwd_kernel` 31.6 ms / 280 calls, `_bwd_dkdv_kernel`
22.9 / 140, `_bwd_dq_kernel` 14.3 / 140, `_delta_kernel` 3.0 / 140); GEMM 43 %, convolution 17 %, elementwise 10 %, copy 7 %,
norm 4 %. The attention kernels are filed under `other` in the report because their names do not match its attention pattern.


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