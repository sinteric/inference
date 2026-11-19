# serve-sim: the models inside a web server

`serve-sim serve` is an axum server on a multi-threaded tokio runtime that
loads EmbeddingGemma 2, Vela 2.0 and the PP-OCR detector and recognizer in-process
through the public `sinteric-inference` API, the way an application embeds
them. `serve-sim load` drives it and prints one JSON summary.

```bash
export ONEDNN_STATIC_ROOT=$HOME/.cache/sinteric-sdk/onednn-static-tp
cargo build --release -p serve-sim
B=target/release/serve-sim

$B serve --threads 4 --eg2 $EG2 --vela $VELA --ocr $OCR &
$B load --embed 4 --vela 2 --ocr 1 --pages lists/sin100_w3.jsonl \
  --pages-root /path/to/pages --work-us 2000 --duration 60
```

## Server

| Endpoint | What it does |
|---|---|
| `POST /embed` `{"texts": [...], "kind": "query"\|"document"}` | `EmbeddingGemma2::embed` |
| `POST /vela` (SystemOne body) | `Vela2::system_one` |
| `POST /ocr` (PNG/JPEG bytes) | decode + `Detector::detect` + `Recognizer::recognize`, lines scoring under 0.5 dropped (one shared backend in process, one worker each with `--isolate`) |
| `GET /health` | returns at once: is the async side responsive? |
| `GET /work?us=N` | N µs of CPU on the async worker: the host's own handler work |
| `GET /stats` | RSS (anonymous/file), peak RSS, glibc heap (live, free, mmap), threads, CPU time, per-model in-flight/served |
| `POST /admin/unload/{eg2,vela,ocr}` | take the model out (new calls get 503), wait for in-flight calls, `unload()` |
| `POST /admin/load/{eg2,vela,ocr}` | load it again |
| `POST /admin/trim` | `release_free_memory()` with the models loaded; RSS and heap before and after |

Every inference response carries a digest of its output bits and a server
side `timing` split: `queue_ms` (handler entry until the blocking call
starts) and `infer_ms` (the call, including waits on the library's locks).

Knobs: `--threads`, `--exec-slots`, `--cpuset`, `--pin-process`,
`--cpu-isa` and `--workspace-limit-mib` map to `RuntimeOptions`.
`--compute-order fifo|any` picks the compute lock order
(`rt_threads::set_compute_order`). `--tokio-workers` and `--max-blocking`
size tokio. `--max-inflight N` puts a per-model semaphore in front of the
blocking pool. `--dispatch inline` calls the models on the async worker
instead of `spawn_blocking`, to measure what that costs the rest of the
server.

`serve-sim split` runs one `serve` child process per model behind a front
server that proxies `/embed`, `/vela`, `/ocr` and `/admin/*` over loopback
HTTP and sums the children's counters in `/stats`. Each child gets its own
`--{eg2,vela,ocr}-threads` and, with `--{eg2,vela,ocr}-cpuset`, is pinned
there (`--pin-process`), so the models run in parallel instead of taking
turns on one pool:

```bash
$B split --eg2 $EG2 --eg2-threads 1 --eg2-cpuset 0 \
  --vela $VELA --vela-threads 2 --vela-cpuset 1-2 \
  --ocr $OCR --ocr-threads 1 --ocr-cpuset 3 &
```

## Load generator

1. `warmup`: one sequential pass over every input; its digests are the
   reference.
2. `serial`: a second sequential pass: solo latency, and the digests must
   match.
3. `idle`: `/health` (and `/work`) probes alone, the server's own baseline.
4. `load`: `--embed/--vela/--ocr N` closed-loop clients for `--duration`
   seconds, with the probes still running at `--probe-hz`. `--churn MODEL`
   unloads and reloads that model every `--churn-every` seconds.
5. `after`: a last sequential pass against the reference, then
   `/admin/trim`.

The summary's `proc` block has `/stats` snapshots at start (models loaded),
after the serial pass and at the end, the trim result, and an RSS / live
heap series over the load phase.

Every response under load is checked against the reference digest for its
input, so cross-request interference shows up as `digest_mismatches`.
`--out FILE` writes every request as JSONL.

Inputs are built in (`src/inputs.rs`): EG2 batches of 1 to 8 texts and
Vela states of 1 to ~400 words, Korean and English. OCR pages come from a
gate list (`--pages`, the `sin100_w3.jsonl` format), the first
`--max-pages` non-warmup entries.

## Results (2026-10-19)

Ryzen 7 9700X (Zen 5), 4 vCPU, `--threads 4`, 45 s loads, the load generator on the
same box. All three models loaded (EG2, Vela 2.0, OCR det+rec). Latency in
ms.

Every run, every configuration: 0 digest mismatches under load and after
it, including across unload/reload.

One model, more clients. Compute busy (Σ rps × solo mean) is 0.91 to 1.01
in every run: the process serves one call at a time, so more clients only
add queueing.

| Model | Clients | rps | p50 | p99 |
|---|---|---|---|---|
| EG2 | 1 | 13.5 | 24 | 276 |
| EG2 | 4 | 12.4 | 223 | 1508 |
| Vela | 1 | 9.6 | 59 | 296 |
| Vela | 4 | 9.7 | 404 | 902 |
| OCR | 1 | 1.25 | 796 | 1065 |
| OCR | 3 | 1.26 | 2375 | 2791 |

Mix: 4 EG2 + 2 Vela + 1 OCR clients.

| Config | EG2 rps / p50 / p99 | Vela rps / p50 / p99 | OCR rps / p50 | RSS after serial → peak (MiB) |
|---|---|---|---|---|
| default | 1.72 / 1664 / 10494 | 5.46 / 310 / 764 | 0.35 / 3088 | 2647 → 3108 |
| `--exec-slots 4` | 7.61 / 377 / 2425 | 2.32 / 812 / 1571 | 0.18 / 5964 | 2668 → 3339 |
| `--max-inflight 1` | 4.09 / 865 / 1590 | 4.64 / 402 / 891 | 0.26 / 3939 | 2671 → 2894 |
| `--threads 3` | 2.32 / 1280 / 7548 | 3.77 / 454 / 1363 | 0.39 / 2730 | → 3085 |
| `--dispatch inline --tokio-workers 2` | 5.89 / 697 / 1338 | 2.85 / 710 / 1340 | 0.37 / 2733 | |

- The operator lock is not fair: by default EG2's short calls lose to
  Vela's, and a 24 ms query waits up to 10 s. `--exec-slots 4` moves the
  starvation onto Vela and OCR. A per-model semaphore of 1
  (`--max-inflight 1`) gives the most even split and the lowest RSS.
- With `spawn_blocking`, `/health` stays at p99 3.8 ms and `/work` at
  4.7 ms under the full mix. Inline dispatch stalls the async workers:
  `/health` p50 714 ms, p99 1459 ms.
- `--threads 3` (one core left to the host) costs 6 to 8% solo latency and
  keeps `/work` at p99 3.0 ms.
- RSS: ~1.5 GB with the three models loaded, 2.65 GB after one serial
  pass, 3.0 to 3.3 GB under concurrent load. It does not come back down
  after the load. Threads 9 → 17 (pool, tokio, blocking pool).
- Churn (`--churn vela --churn-every 12`, 3 cycles): drain 400 to 600 ms,
  RSS 3.0 → 1.7 GB on unload, reload ~1.6 s back to ~2.9 GB. Vela answers
  503 for about 2 s per cycle; EG2 and OCR keep serving and every digest
  matches after the reload.
- Under `taskset -c 2,3` the pool used to pin its workers to CPUs 0 and 1,
  outside the host's mask. It now stays on 2 and 3 (`rt-threads`).

## Fair lock, one process per model, memory (2026-10-19)

Same box and mix (4 EG2 + 2 Vela + 1 OCR clients, 45 s), latency in ms.
0 digest mismatches in every run.

Compute lock order, one process, `--threads 4`:

| Config | EG2 rps / p50 / p99 | Vela rps / p50 / p99 | OCR rps / p50 / p99 |
|---|---|---|---|
| `--compute-order any` (before) | 1.75 / 1653 / 9802 | 5.53 / 304 / 748 | 0.33 / 3361 / 3834 |
| `fifo` (default now) | 4.16 / 679 / 4213 | 1.86 / 910 / 2146 | 0.53 / 1974 / 2381 |
| `fifo --max-inflight 1` | 2.49 / 1404 / 2730 | 2.49 / 651 / 1539 | 0.68 / 1513 / 2061 |

- In arrival order each waiting call gets the next operator, so the models
  share the pool instead of Vela's short operators winning every release.
  The process still computes one operator at a time (Σ rps × solo mean
  0.91 to 0.92 in both).
- One caller pays nothing: EG2 alone with 1 client, 13.7 rps, p50 24.8.

One process per model, `serve-sim split` (`threads@cpus` per child):

| Split | EG2 rps / p50 / p99 | Vela rps / p50 / p99 | OCR rps / p50 / p99 | Peak RSS (MiB) |
|---|---|---|---|---|
| in-process, `fifo` | 4.16 / 679 / 4213 | 1.86 / 910 / 2146 | 0.53 / 1974 / 2381 | 3148 |
| EG2 1@0, Vela 1@1, OCR 2@2-3 | 5.29 / 142 / 3207 | 3.22 / 460 / 1216 | 0.89 / 1099 / 1436 | 3660 |
| EG2 1@0, Vela 2@1-2, OCR 1@3 | 5.20 / 143 / 3194 | 5.65 / 261 / 687 | 0.71 / 1372 / 1920 | 3621 |
| 4 threads each, no cpuset | 5.10 / 517 / 3864 | 4.28 / 346 / 1017 | 0.48 / 2010 / 2781 | 3310 |

- With their own cores the models run in parallel: the process uses 3.9
  cores instead of 3.0. Each call is slower alone (EG2 mean 149 vs 72 on
  one thread, OCR p50 1018 vs 780 on one), so split by expected load.
- Pools that share all four cores interfere and do worse than a partition.
- Cost: about 500 MiB more RSS (three runtimes, heaps and tokio runtimes).

Where the memory goes (one process, all three models):

| Stage | RSS | Live heap | Freed heap kept |
|---|---|---|---|
| models loaded | 1494 | 892 | 574 |
| after one call per input | 2694 | 1864 | 673 |
| end of concurrent load | 3026 | 1654 | 1213 |

- Loading leaves 574 MiB of freed heap resident, nearly all from Vela (539
  MiB in its own process). `Runtime::load` now calls
  `release_free_memory()`: loaded RSS 1494 → 972 MiB, peak 3148 → 2682,
  same latency.
- The first call of each input shape adds about 970 MiB of live heap, which
  then stays flat. Per model, loaded → after use, with what the backend
  counters (`/stats` `memory`) attribute:

  | Model | Live heap (MiB) | Counted after use |
  |---|---|---|
  | EG2 | 375 → 669 | activation pool 221, 210 cached plans |
  | Vela | 500 → 531 | 111 cached plans |
  | OCR | 16 → 665 | conv arena 209 (last page's size), conv weights 8, 726 cached plans |

  The rest, about 430 MiB for OCR and 70 MiB for EG2, is oneDNN primitive
  state behind the cached plans (scratchpads are already user-owned). The
  plan byte counters added with the memory budget (below) measure it: OCR's
  convolution plans 399 MiB and its matmul plans 35 MiB, EG2's plans 65
  MiB, Vela's 29 MiB. OCR builds plans per page and line width
  (`RecConfig.plan_capacity`, 512), so it holds the most.
- `--workspace-limit-mib 64` only trims the activation pool: live heap after
  the serial pass 1864 → 1706, peak RSS 2682 → 2604, same latency.
- Under concurrent load live heap stays flat but RSS grows another 300 to
  500 MiB: freed memory held in glibc's per-thread arenas.
  `MALLOC_ARENA_MAX=2` cut peak RSS 3148 → 2658 with the same latency.
- A 180 s run plateaus (2.6 to 2.8 GiB): no leak.

## Memory budget (2026-10-19)

`--memory-budget-mib N` sets `RuntimeOptions::memory_budget_bytes`, the
same budget for each model. Same mix and box, one process, `--threads 4`,
latency in ms, RSS in MiB. 0 digest mismatches in every run.

| Budget per model | EG2 rps / p50 / p99 | Vela rps / p50 / p99 | OCR solo p50, rps | RSS after serial / end / peak | OCR evictions |
|---|---|---|---|---|---|
| none | 4.22 / 666 / 4225 | 1.85 / 930 / 2095 | 777, 0.54 | 2219 / 2568 / 2677 | 0 |
| 768 | 4.30 / 664 / 4155 | 1.88 / 878 / 2692 | 759, 0.55 | 2220 / 2767 / 2754 | 0 |
| 512 | 4.29 / 675 / 4148 | 1.87 / 912 / 2009 | 801, 0.52 | 1926 / 2180 / 2326 | 34 |
| 512, `--malloc-arena-max 2` | 4.32 / 672 / 4204 | 1.85 / 902 / 2240 | 834, 0.51 | 1890 / 1984 / 2208 | |
| 256 | 5.26 / 499 / 3347 | 2.39 / 722 / 1942 | 2143, 0.17 | 1686 / 1625 / 1970 | 1347 |

- What each model keeps with no budget: OCR 645 MiB (convolution plans
  399, matmul plans 35, arena 209 at the largest page), EG2 about 300
  (pool up to 221, plans 65), Vela 30 to 90. At 256 EG2 pools fewer
  activation buffers and neither it nor Vela evicts a plan, so OCR is the
  model the budget cuts in these runs.
- At 512 OCR first cuts its arena back to each page's size, then evicts
  the plans it used least recently: 34 evictions over the run, OCR 3 to
  7% slower, peak RSS 350 to 470 MiB lower. With `malloc_arena_max 2`
  the end RSS is 580 MiB under the unbudgeted run.
- Below OCR's working set (about 435 MiB of plans for these 100 pages)
  every page rebuilds plans another page evicted: at 256 OCR is 2.8x
  slower. A `budget_evictions` count that keeps rising with calls in
  `backend_stats()` means the budget is under the working set.
- A first run that counted only matmul plans left OCR's convolution
  plans out of the budget; counting them without trimming the arena
  first thrashed already at 512 (OCR 2.6x slower). Both are fixed above.

## Worker processes (2026-10-19)

`serve --isolate` runs each model in a `worker::Isolated` child process
(the same binary, re-run), with `--{eg2,vela,ocr}-threads`, `-cpuset` and
`-budget-mib`. Same mix, EG2 1@0, Vela 2@1-2, OCR 1@3 for both multi-process
rows; RSS is the sum over the whole process tree, sampled each second (MiB).
0 digest mismatches in every run.

| Layout | EG2 rps / p50 / p99 | Vela rps / p50 / p99 | OCR rps / p50 / p99 | `/health` p99 | Peak tree RSS |
|---|---|---|---|---|---|
| in-process, `--threads 4` | 4.17 / 675 / 4218 | 1.84 / 926 / 2354 | 0.53 / 1998 / 2398 | 17.7 | 2787 |
| `split` (HTTP to `serve` children) | 5.30 / 138 / 3195 | 5.58 / 265 / 1027 | 0.72 / 1341 / 1862 | 5.7 | 3183 |
| `serve --isolate` (workers) | 5.37 / 145 / 3194 | 5.65 / 244 / 620 | 0.71 / 1376 / 1890 | 5.4 | 3216 |

- Workers give split's throughput from the library, behind one server:
  the host process peaks at 198 MiB and the models' memory is in the
  workers (at ready: EG2 413, Vela 542, OCR 45 MiB).
- The socket costs nothing measurable: solo p50 EG2 24.6 vs 24.4 ms, OCR
  1018 vs 1008 against split's HTTP children.
- A worker call blocks like any model call. A first run collected `/stats`
  (a round trip to each worker that waits behind its calls in flight) on
  the async worker: `/health` p99 287 ms. Collecting it in
  `spawn_blocking` fixed it.
- `tests/worker.rs` covers bit-equal results against in-process, a worker
  killed mid-call (`Error::Worker`, the next call starts a new worker),
  typed errors across the boundary, a per-worker budget, and unload ending
  the process.
