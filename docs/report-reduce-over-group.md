# Report draft: `sycl::reduce_over_group` segfaults the host process

*Ready to paste into a vendor tracker. Everything in it is measured on this
machine; the reproducer is `debug/native/reproduce_reduce_over_group.cpp`.*

---

**Summary.** A `sycl::reduce_over_group()` over a work-group killed the *host*
process — SIGSEGV, and once SIGBUS — with no SYCL exception, no diagnostic
message and no core dump that pointed anywhere. The crash has since stopped
reproducing (about 35 000 reductions across two toolchains on the current
driver), so this is a report of a bug that silently vanished rather than a
reproducer you can run today. The reduced construct is also 16 % faster than
the form we had to fall back to, so we would like to know what happened and
whether it can come back.

**Environment**

| | |
|---|---|
| GPU | Intel Arc B580 (Xe2, 12.2 GB) |
| Driver | 1.17.39758 (the crashing runs were on an earlier driver; the version is not recorded) |
| OS | Linux, kernel 6.x, x86-64 |
| Toolchains | oneAPI 2025.3 and 2026.0 (DPC++ `icpx`, `-O3 -fsycl -std=c++17`) |
| SYCL implementation | the one shipped with the toolchain, Level Zero backend |

**What happens.** A `parallel_for` over an `nd_range<1>` whose work-item calls
`sycl::reduce_over_group(it.get_group(), value, sycl::maximum<>())` returns
normally while the process is alive, and then the process dies with SIGSEGV —
inside the driver, not in user code, and with no exception thrown by SYCL. The
work-item writes its result to a buffer first, so the reduction is not
optimised away. The last line printed before the death names the case that
died.

**Reproducer.** `debug/native/reduce_over_group/reproduce_reduce_over_group.cpp`:

```bash
icpx -O3 -fsycl -std=c++17 reproduce_reduce_over_group.cpp -o reproduce_reduce_over_group
./reproduce_reduce_over_group
```

Exit status: `0` = every case completed and matched the CPU result;
`139` (128+SIGSEGV) = the crash reproduced; `1` = a result was wrong (not a
crash, but also a bug). The file prints one line per case *before* the case
runs and flushes, so the last line on screen names the case that died.

The matrix: `maximum<float>`, `minimum<int32_t>`, `plus<float>`,
`bit_or<uint32_t>`; work-group sizes 32, 64, 128, 192, 256; a global range of
2^16 rounded up to a multiple of the work-group size; one reduction per case
per iteration.

**How often.** Three crashes in total, all of them while timing a *production*
kernel of ours that used this construct for a per-row maximum. Two SIGSEGV and
one SIGBUS. After the driver update, the same reproducer ran about 35 000
reductions (all four operations, four work-group sizes, both toolchains,
200 iterations each) without a single failure or a wrong result.

**Why we are reporting it anyway.** The failure mode is a killed process, so
no amount of runtime checking can defend against it: our verification
machinery compares results, and a segfault is not a wrong result, it is no
result. We therefore cannot ship the faster construct, and we are not able to
tell whether the bug is fixed or merely dormant.

**The workaround, and what it costs.** We compute the same per-row maximum with
a local-memory reduction followed by a butterfly over the work-group
registers. It is bit-identical to the crashing form and 16 % slower than it
(0.836x the time, interleaved A/B on the full kernel: an FFN epilogue that
takes 0.774 ms becomes 0.925 ms).

**What we would like to know.**

1. Was this a known defect, and is there a release note or an erratum for it?
2. Is the current driver version known to contain the fix, or is the crash
   still latent (i.e. it needs an older driver, a different GPU, a different
   work-group size, or a much longer run to hit)?
3. Is there a way to get a diagnostic out of the runtime when this happens —
   for example a way to have the SYCL runtime check the reduction's
   preconditions and throw instead of corrupting the process?

**Attached.** `reproduce_reduce_over_group.cpp` (the reproducer above) and the
pair of timings (the crashing construct and the butterfly form) for the FFN
epilogue.

---

## Notes for us, not for the tracker

* **Where it goes.** Not to a repository of ours: this is an Intel defect, and
  the two plausible destinations are
  * the **Intel® oneAPI DPC++/C++ Compiler** board --
    <https://community.intel.com/t5/Intel-oneAPI-DPC-C-Compiler/bd-p/oneapi-data-parallel-c>
    -- which has a thread made for exactly this, *"Intel DCP++ Crash report -
    priority support"* --
    <https://community.intel.com/t5/Intel-oneAPI-DPC-C-Compiler/Intel-DCP-Crash-report-priority-support/td-p/1590087>;
  * the **Intel GPU Community Issue Tracker** on GitHub --
    <https://github.com/IGCIT/Intel-GPU-Community-Issue-Tracker-IGCIT> --
    which is the public tracker for the GPU driver, and is where a *driver*
    fault belongs.

  Which one is right depends on where their triage says the fault is. The
  crash kills the *host* process, so the fault is in host-side code: the
  oneAPI runtime (`libsycl.so`, closed source), the Level Zero loader
  (`libze_loader.so`, open source --
  <https://github.com/oneapi-src/level-zero>), or the GPU driver itself
  (closed source). Start with the compiler board, mention the driver tracker,
  and let Intel route it. If they ask for a GitHub issue, the IGCIT tracker is
  the public one.

* The reproducer is built by `debug/native/build.sh` (it is the last target
  there) and lives in the public repository, so the tracker can fetch it
  directly: <https://github.com/okolenmi/oko-xpu-kernels>.
* The crash needs a *driver* from before 1.17.39758, which is why the report
  says the version is not recorded. If the tracker asks, the honest answer is
  "an earlier 1.17.x driver, and the machine has been updated since".
* Re-test after every driver or compiler bump:
  `PROBE_REDUCE=1 ./debug/native/bin/probe_ffn_situ 3973 8192 device`
  (the probe keeps the `reduce_over_group` variant behind that flag).
