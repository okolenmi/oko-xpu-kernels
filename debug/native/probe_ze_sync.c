/*
 * LD_PRELOAD interposer on the Level Zero host-wait entry points.
 *
 * The warm-forward profile shows ~96 host waits per forward inside the
 * driver, and the GPU idles while the host catches up.  The torch profiler
 * does not unwind through the driver, so the caller has to be found at the
 * ABI boundary: this interposes the symbols, prints a backtrace and
 * forwards the call.
 *
 *   gcc -shared -fPIC -o probe_ze_sync.so probe_ze_sync.c -ldl
 *   LD_PRELOAD=./probe_ze_sync.so python ComfyUI/main.py ...
 *
 * Set PROBE_ZE_SYNC_LIMIT to stop printing after N calls (the count is
 * still reported at exit).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*sync_fn)(void *);
static long counter;
static long limit = 200;
static int in_hook;

static void *resolve_next(const char *name) {
    void *p = dlsym(RTLD_NEXT, name);
    if (!p) {
        fprintf(stderr, "probe_ze_sync: cannot resolve %s\n", name);
        exit(2);
    }
    return p;
}

static void trace(const char *name) {
    counter++;
    if (!in_hook && counter <= limit) {
        in_hook = 1;
        void *bt[20];
        int n = backtrace(bt, 20);
        fprintf(stderr, "=== %s #%ld ===\n", name, counter);
        backtrace_symbols_fd(bt, n, 2);
        in_hook = 0;
    }
}

__attribute__((destructor)) static void report(void) {
    fprintf(stderr, "=== probe_ze_sync: %ld host waits ===\n", counter);
}

int zeCommandListHostSynchronize(void *h) {
    static sync_fn real;
    if (!real) real = (sync_fn)resolve_next("zeCommandListHostSynchronize");
    trace("zeCommandListHostSynchronize");
    return real(h);
}

int zeEventHostSynchronize(void *h) {
    static sync_fn real;
    if (!real) real = (sync_fn)resolve_next("zeEventHostSynchronize");
    trace("zeEventHostSynchronize");
    return real(h);
}

int zeFenceHostSynchronize(void *h) {
    static sync_fn real;
    if (!real) real = (sync_fn)resolve_next("zeFenceHostSynchronize");
    trace("zeFenceHostSynchronize");
    return real(h);
}

__attribute__((constructor)) static void init(void) {
    const char *e = getenv("PROBE_ZE_SYNC_LIMIT");
    if (e) limit = atol(e);
}
