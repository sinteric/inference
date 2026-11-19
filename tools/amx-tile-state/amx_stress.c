// Raw AMX-BF16 stress, no oneDNN: N threads, each repeatedly runs a tile GEMM
// with fixed inputs and compares against the first result. Detects hardware /
// hypervisor / kernel tile-state corruption under preemption.
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18
typedef struct { uint8_t pal, start; uint8_t rsv[14]; uint16_t colsb[16]; uint8_t rows[16]; } __attribute__((packed)) cfg_t;
static int iters, yield_every, release_every;
static long bad_total;
static void *run(void *arg) {
  long id = (long)arg;
  cfg_t cfg; memset(&cfg, 0, sizeof cfg); cfg.pal = 1;
  for (int i = 0; i < 8; i++) { cfg.colsb[i] = 64; cfg.rows[i] = 16; }
  uint16_t *A = aligned_alloc(64, 16*64*8), *B = aligned_alloc(64, 16*64*8);
  float *C = aligned_alloc(64, 16*64*4), *R = aligned_alloc(64, 16*64*4);
  for (int i = 0; i < 16*32*4; i++) { A[i] = 0x3f80 + (i * 7 % 64); B[i] = 0x3f00 + (i * 13 % 64); }
  _tile_loadconfig(&cfg);
  long bad = 0;
  for (int it = 0; it < iters; it++) {
    _tile_zero(0); _tile_zero(1);
    for (int kk = 0; kk < 8; kk++) {
      _tile_loadd(2, A + kk*16*32, 64);
      _tile_loadd(3, B + kk*16*32, 64);
      _tile_dpbf16ps(0, 2, 3);
      _tile_dpbf16ps(1, 2, 3);
    }
    _tile_stored(0, C, 64);
    if (it == 0) memcpy(R, C, 16*64);
    else if (memcmp(R, C, 16*64)) bad++;
    if (yield_every && it % yield_every == 0) sched_yield();
    if (release_every && it % release_every == 0) { _tile_release(); _tile_loadconfig(&cfg); }
  }
  _tile_release();
  __atomic_fetch_add(&bad_total, bad, __ATOMIC_RELAXED);
  return 0;
}
int main(int argc, char **argv) {
  int nthr = atoi(argv[1]); iters = atoi(argv[2]);
  yield_every = argc > 3 ? atoi(argv[3]) : 0; release_every = argc > 4 ? atoi(argv[4]) : 0;
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) { perror("arch_prctl"); return 2; }
  pthread_t t[64];
  for (long i = 0; i < nthr; i++) pthread_create(&t[i], 0, run, (void*)i);
  for (int i = 0; i < nthr; i++) pthread_join(t[i], 0);
  printf("threads=%d iters=%d yield=%d release=%d bad=%ld\n", nthr, iters, yield_every, release_every, bad_total);
  return bad_total != 0;
}
