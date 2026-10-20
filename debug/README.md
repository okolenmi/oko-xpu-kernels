# debug/ — measurement instruments

These are the probes that produced every number in `docs/performance.md`.
They live here rather than in `/tmp` because they are the evidence: if someone
disagrees with a conclusion, they should be able to re-run the experiment.

## Setup

```bash
export PATH=/home/okolenmi/.local/oneapi/opt/intel/oneapi/compiler/2026.0/bin:$PATH
./build.sh                  # builds build/libxpuint8.so (needed by the .py probes)
./debug/native/build.sh     # builds the standalone SYCL probes into debug/native/bin
```

`debug/native/build.sh` also puts the DPC++ runtime on `LD_LIBRARY_PATH`
itself, since there is no root and no system-wide install.

## Python probes (need `build/libxpuint8.so`)

| script | question it answers | headline answer |
|---|---|---|
| `bench_stages.py` | where does end-to-end time go? | rotate 0.52 / `_int_mm` 0.50 / epilogue 0.15 ms at M=4096 |
| `bench_bf16_fair.py` | **does INT8 beat BF16 when both pay for the rotation?** | INT8 is 5-15% faster for M>=1024, loses at M=512 |
| `verify_rotate_scales.py` | is the rotation numerically right? | matches float64 to 6.4e-08; eager bf16 bmm is up to 40% wrong |
| `probe_gemm_shapes.py` | why is the hand-written GEMM only 43 TOPS? | L2-bandwidth bound at ~4 TB/s, not DPAS bound |
| `probe_intmm_layout.py` | is `torch._int_mm` layout- or packing-sensitive? | no; `_weight_int8pack_mm` needs a float A so is not an int8xint8 path |
| `probe_n_blocking.py` | can N-blocking keep the int32 accumulator in L2? | no — blocked `_int_mm` costs 2x what it saves |
| `probe_alloc_cost.py` | how expensive are the pipeline temporaries? | `torch.empty` is 0.004 ms regardless of size |
| `probe_load_ab.py` | paired A/B of two library builds | the method to use for any small effect; the GPU drifts 10-30% with thermals |

`bench_bf16_fair.py` is the one that matters most: the original
`benchmarks/bench_pipeline.py` compares INT8 against a BF16 `linear()` that
omits the activation rotation, which is not a fair comparison — ConvRot needs
the rotation on both paths.

## Native probes (standalone SYCL, no torch)

Run any of them as `./debug/native/bin/<name> [args]`.

| probe | question | headline answer |
|---|---|---|
| `probe_rotate_ablation` | which part of the rotate costs what? | load+store 0.273, transform 0.134, abs-max 0.025 ms |
| `probe_bandwidth` | what are the real DRAM ceilings? | ~310 GB/s read / ~330 GB/s write at 25 MB |
| `probe_epilogue` | is the scalar bf16 convert costing anything? | no — 88% of the bandwidth floor |
| `probe_lane_utilisation` | do the 4 idle lanes at K=3072 cost 25%? | no, it's a cache-size effect |
| `probe_bf16_rotate` | best store pattern for the BF16 rotate? | flat store loop, 1.44x faster than interleaved |
| `probe_load_store_width` | do 64-byte loads help? | no |

### rejected/

Experiments that did **not** work. They are kept because each one closes off a
plausible-looking optimisation, and several of them were only rejected after
real work:

- `rotate_variants` — fp16 butterfly: **2x slower**
- `rotate_shared_memory` — staging the strided axis through LDS: **2.4x slower**
- `rotate_three_pass` — rotate+stage / reduce / quantize: **1.4x slower**
- `rotate_workgroup_size` — work-group 16..256: flat, not a lever
- `rotate_pack_and_width` — int8 pack and store width: no difference
- `conversion_costs` — also records the SYCL type limits (no 64-wide `vec`,
  no free `sycl::convert`) that force the scalar pack loops
- `simd_width_check` — confirms `vec<float,16>` really is 512-bit and near peak

## Timing discipline

The B580 is noisy enough that a single mean over 20 iterations is not
trustworthy — the same binary has measured 0.52 and 0.63 ms for the same call.
Everything here takes **min over several reps of mean over N**, via
`common.bench`.

One artefact to be aware of: a kernel that repeatedly writes the *same* output
buffer measures faster than one writing a fresh buffer, because the first keeps
the buffer L2-warm. The dequant epilogue is 0.135 ms writing one fixed buffer
and ~0.275 ms writing a fresh one. Real models rotate buffers, so 0.275 ms is
the honest number.

## Gotcha that cost real time

`ctypes` defaults unannotated integer arguments to a 32-bit C `int`. If you
call a library function taking `void*` without setting `argtypes`, a 64-bit
`tensor.data_ptr()` is silently truncated, the kernel writes through garbage,
and the failure surfaces as `UR_RESULT_ERROR_DEVICE_LOST` — not as an argument
error. Always go through `common.load_lib()`, which declares everything.