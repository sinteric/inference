// AMX tile-state probe. Some virtual machines lose the AMX TILEDATA state of a
// thread that is preempted while its tiles are live: the thread resumes with
// zeroed tiles and silently computes wrong GEMM blocks. This runs a fixed
// tile GEMM on one CPU while a second thread repeatedly preempts it, and
// counts results that differ from the first.
//
// The tile code uses per-function target attributes, so no AMX compiler flags
// are needed and the file builds on any target.
#include <cstdint>

extern "C" {

struct snt_amx_probe_result {
    int status; // 0 = ran, -1 = AMX unavailable (CPU, OS permission or arch)
    uint64_t iterations; // tile GEMMs checked
    uint64_t corrupt; // results that differed from the first
    uint64_t preemptions; // involuntary context switches of the checked thread
};

}

// Without x86-64 Linux or a compiler that ships the AMX intrinsics (gcc >= 11,
// clang >= 12) the probe is a stub that reports status -1.
#if defined(__x86_64__) && defined(__linux__) && defined(__has_include)
#if __has_include(<amxtileintrin.h>)
#define SNT_AMX_PROBE 1
#endif
#endif

#if defined(SNT_AMX_PROBE)

#include <cpuid.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <cstring>

namespace {

constexpr int ARCH_REQ_XCOMP_PERM = 0x1023;
constexpr int XFEATURE_XTILEDATA = 18;

struct TileCfg {
    uint8_t palette;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __attribute__((packed));

uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

bool cpu_has_amx() {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    return (d & (1u << 24)) && (d & (1u << 22)); // AMX-TILE, AMX-BF16
}

struct Shared {
    int cpu;
    uint64_t deadline_ns;
    snt_amx_probe_result *out;
};

// Alternates short bursts of work with short sleeps so the scheduler keeps
// preempting the checked thread on the shared CPU (wake-up preemption).
void *contender(void *p) {
    auto *s = static_cast<Shared *>(p);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(s->cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
    while (now_ns() < s->deadline_ns) {
        uint64_t spin_until = now_ns() + 30000;
        while (now_ns() < spin_until) {
        }
        timespec ts = {0, 50000};
        nanosleep(&ts, nullptr);
    }
    return nullptr;
}

__attribute__((target("amx-tile,amx-bf16"), noinline)) void
stress(Shared *s) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(s->cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);

    alignas(64) uint16_t a[8 * 16 * 32], b[8 * 16 * 32];
    alignas(64) float c[16 * 16], ref[16 * 16];
    for (int i = 0; i < 8 * 16 * 32; i++) {
        a[i] = uint16_t(0x3f80 + (i * 7 % 64));
        b[i] = uint16_t(0x3f00 + (i * 13 % 64));
    }
    TileCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette = 1;
    for (int i = 0; i < 8; i++) {
        cfg.colsb[i] = 64;
        cfg.rows[i] = 16;
    }
    _tile_loadconfig(&cfg);
    uint64_t iters = 0, corrupt = 0;
    while (true) {
        _tile_zero(0);
        for (int k = 0; k < 8; k++) {
            _tile_loadd(2, a + k * 16 * 32, 64);
            _tile_loadd(3, b + k * 16 * 32, 64);
            _tile_dpbf16ps(0, 2, 3);
        }
        _tile_stored(0, c, 64);
        if (iters == 0) {
            memcpy(ref, c, sizeof(c));
        } else if (memcmp(ref, c, sizeof(c)) != 0) {
            corrupt++;
        }
        iters++;
        if ((iters & 255) == 0 && now_ns() >= s->deadline_ns) break;
    }
    _tile_release();
    rusage ru;
    getrusage(RUSAGE_THREAD, &ru);
    s->out->iterations = iters;
    s->out->corrupt = corrupt;
    s->out->preemptions = uint64_t(ru.ru_nivcsw);
}

void *stress_entry(void *p) {
    stress(static_cast<Shared *>(p));
    return nullptr;
}

} // namespace

extern "C" snt_amx_probe_result sinteric_amx_tile_state_probe(int budget_ms) {
    snt_amx_probe_result r = {-1, 0, 0, 0};
    if (!cpu_has_amx()) return r;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0)
        return r;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return r;
    int cpu = -1;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &allowed)) {
            cpu = i;
            break;
        }
    }
    if (cpu < 0) return r;
    Shared s = {cpu, now_ns() + uint64_t(budget_ms < 1 ? 1 : budget_ms) * 1000000ull, &r};
    pthread_t other, checked;
    // Threads are used (not the caller) so the caller's affinity is untouched.
    if (pthread_create(&other, nullptr, contender, &s) != 0) return r;
    if (pthread_create(&checked, nullptr, stress_entry, &s) != 0) {
        s.deadline_ns = 0;
        pthread_join(other, nullptr);
        return r;
    }
    pthread_join(checked, nullptr);
    pthread_join(other, nullptr);
    r.status = 0;
    return r;
}

#else

extern "C" snt_amx_probe_result sinteric_amx_tile_state_probe(int) {
    return {-1, 0, 0, 0};
}

#endif
