# debug/ — the instruments

Every number in `docs/performance.md` came from a script in here. They live in
the repo rather than in `/tmp` because they are the evidence: if someone
disagrees with a conclusion, they should be able to re-run the experiment.

## Setup

```bash
export SYCL_ROOT=/home/okolenmi/.local/oneapi-2025.3/2025.3
./build.sh                  # -> build/libxpuint8.so   (the Python probes need it)
./debug/native/build.sh     # -> debug/native/bin/*    (the standalone SYCL probes)
```

`debug/native/build.sh` puts the DPC++ runtime on `LD_LIBRARY_PATH` itself,
since there is no root and no system-wide install. A standalone probe run by
hand needs:

```bash
export LD_LIBRARY_PATH=$SYCL_ROOT/lib:$SYCL_ROOT/lib/x64
export LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero
```

## The verifiers (the important ones)

These answer "is this correct", and they are the ones to re-run after any
change:

| script | what it proves |
|---|---|
| `verify_linear_dx.py` | `linear_dx` against autograd, and the INT8 weight's own error |
| `verify_linear_dw.py` | `linear_dw` against autograd of the dequantized weight (bit-identical) |
| `verify_capture_guards.py` | the capture refusals, and `Int8Linear.dx` replaying bit-identically |
| `verify_int8_capture.py` | the INT8 forward and both backwards captured and replayed bit-identically |
| `verify_graph_capture.py` | the rope and adaLN forwards *and* backwards captured and replayed bit-identically |
| `verify_rotate_scales.py` | the rotation against float64 (the eager bf16 bmm is up to 40 % wrong) |
| `verify_published_wheel.sh` | the **published** wheel renders the recorded PNG md5 — the download, the install, the gate, and the server restart |
| `verify_situ.py`, `verify_ffn_fusion.py`, `verify_qkv_rope_fuse.py` | the fused paths against the unfused chains |
| `verify_ffn_geglu.py` | SDXL's GEGLU: the fused FFN against the chain a model writes, the half order, and the capture |
| `verify_geglu.py` | the fused bf16 GEGLU (`triton.geglu`): differing values against the chain in both directions, chain vs fused time, bandwidth floor |
| `verify_prepare_fused.py` | the fused weight preparation against the torch chain: how different the weights are, whether the error changes, and the speed |
| `verify_trainable_int8.py` | the straight-through linear: the forward, the weight gradient, the two backward modes, and the capture |
| `verify_quant_lib.py` | the Triton quantizer against the torch chain, within the gate's budget |

## The probes (the ones that measure)

| script | question | headline answer |
|---|---|---|
| `bench_bf16_fair.py` | **does INT8 beat BF16 when both pay for the rotation?** | parity, 15-17 % faster at M=1024; loses at M=512 |
| `bench_stages.py` | where does end-to-end time go? | rotate 0.52 / `_int_mm` 0.50 / epilogue 0.15 ms at M=4096 |
| `bench_ffn_fusion.py`, `bench_qkv_chain.py` | what does each fusion buy? | the FFN's intermediate is 2 GEMMs + 3 memory passes |
| `probe_gemm_shapes.py` | why is the hand-written GEMM only 43 TOPS? | L2-bandwidth bound at ~4 TB/s, not DPAS bound |
| `probe_linear_dx_variants.py` | bf16 dequant or an int8 GEMM for the backward? | 2.08 ms against 2.75 ms — a memory option, not a speed one |
| `probe_rope_variants.py`, `probe_adaln_variants.py` | which schedule? | the fused one, bit-exact, 2.34× / 1.56× |
| `probe_rope_backward.py`, `probe_adaln_backward.py` | the backwards | 14.3× / 6.3× against the eager chain |
| `probe_intmm_layout.py` | is `_int_mm` layout- or packing-sensitive? | no; `_weight_int8pack_mm` needs a float A, so it is not an int8 path |
| `probe_n_blocking.py` | can N-blocking keep the accumulator in L2? | no — blocked `_int_mm` costs 2× what it saves |
| `probe_alloc_cost.py` | how expensive are the temporaries? | `torch.empty` is 0.004 ms regardless of size |
| `probe_load_ab.py` | paired A/B of two library builds | the method to use for any small effect |
| `probe_sdxl_shapes.py` | the INT8-vs-BF16 ratio at **SDXL's** shapes and token counts | 2.5× faster on the FFN's down-projection, 1.5× slower on its up-projection (the int32 accumulator), and host-bound below M≈1024 |
| `probe_sdxl_ffn.py` | the fused GEGLU FFN against the unfused INT8 chain | 1.2-1.5× faster, and at parity with bf16 at the big M |

`bench_bf16_fair.py` is the one that matters most: the original
`benchmarks/bench_pipeline.py` compares INT8 against a BF16 `linear()` that
omits the activation rotation, which is not a fair comparison — ConvRot needs
the rotation on both paths.

The `ab_*.sh` scripts are the A/B harnesses for the ComfyUI-side changes
(rope, tanh, the FFN fusion, the library build); `probe_moss.py` renders the
reference image through the ComfyUI server, and `server.sh` starts it. The
recorded gate is **md5 `a2aacfda45409493a490b0402b61f850`** — seed 1234,
3 steps, `--deterministic`.

## Native probes (standalone SYCL, no torch)

Run any of them as `./debug/native/bin/<name> [args]`.

| probe | question | headline answer |
|---|---|---|
| `probe_rotate_ablation` | which part of the rotate costs what? | load+store 0.273, transform 0.134, abs-max 0.025 ms |
| `probe_bandwidth` | what are the real DRAM ceilings? | ~310 GB/s read / ~330 GB/s write at 25 MB |
| `probe_epilogue` | is the scalar bf16 convert costing anything? | no — 88 % of the bandwidth floor |
| `probe_ffn_situ` | the row max: butterfly or `reduce_over_group`? | `reduce_over_group` is 16 % faster and bit-identical, but it once killed the host — `PROBE_REDUCE=1` |
| `probe_cross_lane` | can the rotate's last open shape be built? | no: the reduction domain, not the exchange |
| `probe_bf16_rotate` | best store pattern for the BF16 rotate? | a flat store loop, 1.44× faster than interleaved |
| `probe_lane_utilisation` | do the 4 idle lanes at K=3072 cost 25 %? | no, it is a cache-size effect |
| `probe_load_store_width` | do 64-byte loads help? | no |
| `probe_tanh_exact` | is `expm1(2x)/(expm1(2x)+2)` bf16-identical to `tanh`? | yes, for every finite bf16 input |
| `probe_gelu_exact` | can the erf-GELU be reproduced bit for bit? | yes: `sycl::erf`'s GELU matches torch's `F.gelu(approximate="none")` for all 65536 bf16 values (the reference is dumped by `debug/probe_gelu_torch.py`) |
| `reproduce_reduce_over_group` | does `reduce_over_group` still kill the process? | not on driver 1.17.39758 — but no fix was ever claimed |

`debug/native/rejected/` holds the experiments that did not work (the fp16
butterfly, the shared-memory staging, the three-pass version, the work-group
sweep). They are kept because each one closes off a plausible-looking
optimisation, and several were rejected only after real work.

## Timing discipline

The B580 is noisy enough that a single mean over 20 iterations is not
trustworthy — the same binary has measured 0.52 and 0.63 ms for the same call.
Everything here takes **min over several reps of mean over N**, via
`common.bench`. Ratios come from *interleaved* rounds in one process.

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
