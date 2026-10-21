# `sycl::reduce_over_group` killed the host process — report to the vendor

**Status: sent** to the oneAPI DPC++/C++ Compiler board and to the Intel GPU
Community Issue Tracker (GitHub). This file is the record of it: the
reproducer, the evidence, and the three questions worth asking. If the text
that was sent differs from this, this file is the summary, not the transcript.

## What happens

`sycl::reduce_over_group(it.get_group(), vm, sycl::maximum<>())` — the
standard work-group reduction — terminated the host process with `SIGSEGV` /
`SIGBUS`, with no SYCL exception and nothing in the error queue. Twice, in
two different kernels, on two different toolchains. The same code path
reduced correctly in every other respect: where it did not crash, the result
was bit-identical to the butterfly reduction it replaced.

## Why it is not shipped

The variant is **16 % faster** and bit-identical:

| row-max form (M=3973, K=8192) | time |
|---|---|
| sub-group maxes to local memory, barrier, serial loop (production) | 0.925 ms |
| `sycl::reduce_over_group` | 0.774 ms |

Interleaved A/B: mode15 / mode0 = **0.836**, range 0.831-0.845, identical
under the 2025.3 toolchain (0.833, 0.830-0.840). 0 of 32 546 816 output bytes
differ.

The win is ~6.4 ms of a ~1050 ms step — 0.6 %, below a step-level slope's
noise floor. Against that: the failure mode is a **killed process**, and a
gate cannot catch that. It is not a wrong number, it is a dead host, so no
amount of verification makes it safe to ship.

## The reproducer

```bash
./debug/native/build.sh
./debug/native/bin/probe_ffn_situ 3973 8192 device     # PROBE_REDUCE=1 for the variant
./debug/native/bin/reproduce_reduce_over_group          # 8 work-group sizes x 4 reduction ops
```

The second one is the isolated reproducer: 8 work-group sizes × 4 reduction
ops × 200 launches × 5 processes, both toolchains, on Level Zero.

## What changed since

On 2026-10-21 the reproducer ran **~35 000 reductions of real work-group
widths** (the epilogue itself uses 512) without a fault, on driver
`1.17.39758`, with both toolchains. That is a statement about this driver on
this machine, **not a fix claim**: no changelog entry was found, and the
crash was never filed upstream before this report.

The production kernels keep the butterfly + local-memory form. The probe keeps
the variant behind `PROBE_REDUCE=1`, so the next driver bump can re-test it in
one command:

```bash
PROBE_REDUCE=1 ./debug/native/bin/probe_ffn_situ 3973 8192 device
```

## The three questions

1. Was this known? A `reduce_over_group` that segfaults the host, with no
   exception, is worth knowing about.
2. Is the fix in the current driver, or was the bug dormant (a code path that
   is only reached at particular work-group widths)?
3. Can the runtime **throw** instead of killing the process? A construct that
   can only fail by taking the process with it is unusable in a library that
   runs on someone else's card.
