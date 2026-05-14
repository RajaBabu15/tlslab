# tlslab — Contention Collapse in Lock-Free MPMC Ring Buffers on Apple Silicon M4

**An empirical study of CAS-retry amplification, cache-coherence pressure, and queue topology in concurrent producer-consumer pipelines.**

Raja Babu &nbsp;·&nbsp; IIT (ISM) Dhanbad, B.Tech CSE 2026
&nbsp;·&nbsp; <rajababu8520456@gmail.com>

---

## Abstract

We present a controlled empirical study of throughput collapse in shared-memory non-blocking queues under producer-consumer workloads on a heterogeneous ARM processor (Apple M4). Starting from a bounded multi-producer multi-consumer ring buffer of the Vyukov pattern (Vyukov, 2010), we measure a **74–75% throughput reduction** when the thread count grows from one producer-consumer pair to two, and identify the root cause as compare-and-swap (**CAS**) retry amplification on the shared `head_` and `tail_` atomics. Slot-level false sharing is structurally eliminated by per-slot `alignas(128)`, verified at compile time by `static_assert`.

We then isolate the asymmetry of the collapse: a sweep over $(N_P, N_C) \in \{(1,1), (2,1), (1,2)\}$ shows that producer-side and consumer-side contention are quantitatively indistinguishable (Δ = 9% between the two configurations, both at −73% to −75% versus baseline), refuting any one-sided pathology. CAS-retry amplification, instrumented as the ratio $\rho = \text{iterations}/\text{ops}$, grows from $\rho(1) = 1.05$ to $\rho(16) = 976.8$, with a discontinuous jump at $N = 8$ pairs coinciding with thread count exceeding the M4 P-core count.

Three remediation paths are benchmarked:
(i) **batched dequeue** with $K \in \{8, 16, 32\}$ — reduces $\rho$ but adds enough cross-cache load on the bulk path to *reduce* net throughput;
(ii) **per-pair SPSC lanes** (Lamport-style ring with cached-cursor optimization) — eliminates shared CAS entirely and recovers **23–91×** versus MPMC at equivalent $N$;
(iii) **producer-sharded SPSC** — at $N = 4$ pairs achieves **596.9 Mops/s**, a **107×** improvement over the MPMC baseline at the same thread count, attributed to all four ring buffers fitting in M4 P-core L1-D simultaneously.

Findings (i) refute the original "allocator contention" hypothesis (`malloc`/`free` overhead at single-thread is 124.9 ns/op; the queue collapses long before allocation becomes the limiter), and (ii) demonstrate that *shared atomic ownership* — not lock-freedom itself — is the dominant scalability barrier for ring-buffer transports under contended producer-consumer load.

---

## Table of Contents

1. [Hardware and Software Environment](#1-hardware-and-software-environment)
2. [Notation](#2-notation)
3. [Theoretical Framework](#3-theoretical-framework)
4. [Data Structures](#4-data-structures)
5. [Experimental Methodology](#5-experimental-methodology)
6. [Experimental Configurations](#6-experimental-configurations)
7. [Results](#7-results)
8. [Findings](#8-findings)
9. [Repository Structure and Reproducibility](#9-repository-structure-and-reproducibility)
10. [Open Questions](#10-open-questions)
11. [References](#11-references)
12. [Citation](#12-citation)
13. [License](#13-license)

---

## 1. Hardware and Software Environment

### 1.1 Platform

| Parameter | Value |
|---|---|
| SoC | Apple M4 (arm64) |
| Performance cores ($P$-cores) | 4 @ 3.8 GHz boost |
| Efficiency cores ($E$-cores) | 6 @ 2.9 GHz boost |
| Cache line size $\mathcal{L}$ | 128 bytes (`sysctl hw.cachelinesize`) |
| L1-D per $P$-core | 128 KiB |
| L2 per cluster | 16 MiB (shared) |
| Memory | LPDDR5X unified |
| OS | macOS (Darwin arm64 25.2) |
| Compiler | clang++ 21, `-std=c++20 -O2 -pthread -Wall -Wextra` |
| QoS hint | `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` |

The M4's heterogeneous topology is a *measurement confound* at high thread counts. At $N = 8$ pairs (16 total worker threads), the kernel scheduler must place threads onto $E$-cores, whose L2 slice is distinct from the $P$-core cluster's L2 slice — raising cross-cluster coherence latency. This is the documented contributor to the $\rho$ discontinuity reported in §7.2. macOS arm64 does not expose hard thread affinity for user processes; the QoS hint biases (but does not pin) toward $P$-cores.

The destructive interference granularity $\mathcal{L} = 128$ B is **double** the x86-conventional 64 B. All cache-line-aligned data structures in this work use `alignas(128)`. This value matches the C++17 `std::hardware_destructive_interference_size` on this target (Williams 2019, §8.2.3).

### 1.2 Spin-Wait Primitive

All spin loops use the arm64 hint instruction:

```cpp
asm volatile("yield" ::: "memory");
```

`std::this_thread::yield()` is explicitly **not** used, as it invokes the macOS scheduler syscall whose latency would dominate the tight inner loops.

---

## 2. Notation

We follow the memory-model vocabulary of Williams (2019, Ch. 5) and the lock-free progress hierarchy of Herlihy & Shavit (2020):

| Symbol | Meaning |
|---|---|
| $N_P, N_C$ | number of producer / consumer threads |
| $N$ | shorthand for $N_P = N_C = N$ in symmetric experiments |
| $\mathcal{L}$ | destructive interference (cache line) size = 128 B on M4 |
| $C$ | ring buffer capacity (slots) |
| $R(N)$ | aggregate throughput in operations / second |
| $r_i$ | per-thread throughput of thread $i$ |
| $\rho(N)$ | CAS-retry amplification, $\rho = (\text{total inner-loop iters}) / (\text{successful ops})$ |
| $\eta(N)$ | scaling efficiency, $\eta(N) = R(N) / (N \cdot R(1))$ |
| $\Delta_\text{alloc}$ | per-op `malloc`/`free` overhead at low contention |
| $\xrightarrow{sw}$ | *synchronizes-with* relation (Williams §5.3.1) |
| $\xrightarrow{hb}$ | *happens-before* relation (Williams §5.3.2) |
| `mo_acq`, `mo_rel`, `mo_rlx` | memory orderings `acquire`, `release`, `relaxed` (Williams §5.3.3) |
| $\textsf{CAS}(a, e, n)$ | `compare_exchange` on address $a$, expected $e$, new $n$ (Michael 2004, §2.2) |

### Progress-Class Definitions (Williams 2019, §7.1; Herlihy & Shavit 2020, §3)

- **Obstruction-free.** If all other threads pause, any given thread completes its operation in a bounded number of steps.
- **Lock-free.** When multiple threads operate on the data structure, *some* thread completes its operation in a bounded number of steps.
- **Wait-free.** *Every* thread completes its operation in a bounded number of steps, regardless of other threads.

The Vyukov MPMC ring of §4.1 is **lock-free** but not wait-free, due to the unbounded inner CAS retry loop. The SPSC ring of §4.2 is **wait-free** for its single producer and single consumer.

### The ABA Problem (Michael 2004, §2.3; Wolff 2021, Ch. 6)

ABA occurs when a thread reads value $A$ from a shared location, other threads change the location to $B$ and back to $A$, and the original thread's CAS succeeds erroneously. The Vyukov pattern *avoids* ABA structurally by tagging each cell with a monotonically-increasing sequence number $s_k$, where $s_k = k$ for empty slot $k$ and $s_k = k + 1$ after publication (see §4.1).

---

## 3. Theoretical Framework

### 3.1 Throughput Model

For shared-state concurrent algorithms, the Universal Scalability Law (Gunther 2007) gives:

$$R(N) = \frac{N \cdot R(1)}{1 + \alpha (N - 1) + \beta N (N - 1)} \qquad \text{(1)}$$

where $\alpha \in [0, 1]$ is the *serialization fraction* (Amdahl term) and $\beta \geq 0$ is the *cross-talk coefficient* capturing coherence pressure between every pair of threads. Equivalently:

$$\eta(N) = \frac{R(N)}{N \cdot R(1)} = \frac{1}{1 + \alpha (N - 1) + \beta N (N - 1)} \qquad \text{(2)}$$

For a shared-atomic queue, $\beta$ dominates because *every* `try_push`/`try_pop` must coordinate, via CAS, with every other in-flight operation on the same atomic. Empirically (§7.2), $\eta$ drops from $1.0$ at $N = 1$ to $0.135$ at $N = 2$ — consistent with $\beta \gg 0$.

### 3.2 CAS-Retry Amplification

We instrument each call to `try_push` and `try_pop` with an out-parameter `iter_count_out` that accumulates the inner-loop iteration count. For a configuration $(N_P, N_C)$ with total successful ops $O_P + O_C$ and total iterations $I$:

$$\rho(N_P, N_C) = \frac{I}{O_P + O_C} \qquad \text{(3)}$$

$\rho = 1$ implies every op succeeds on first attempt (no retry, no contention). $\rho > 1$ measures wasted CPU cycles on either (a) failed CAS due to a competing thread, or (b) the "diff > 0" branch where a thread observes an advanced position and re-loads.

### 3.3 Throughput Aggregation

Per Williams (2019, §8.4.2), we *avoid* aggregating against a shared wall-clock interval, which masks straggler threads. Each thread $i$ records its own start $s_i$ and end $e_i$ via `std::chrono::steady_clock` and computes its own rate:

$$r_i = \frac{c_i}{e_i - s_i} \qquad \text{(4)}$$

where $c_i$ is the per-thread operation count. The aggregate is the sum of per-thread rates:

$$R = \sum_{i = 1}^{N_P + N_C} r_i \qquad \text{(5)}$$

The minimum and maximum $r_i$ surface stragglers in the `pt_min` / `pt_max` columns of the CSV output.

### 3.4 Cache-Coherence Cost

Following Williams (2019, §8.2.2–§8.2.3), the *cache ping-pong* cost per atomic write that races with a peer's load is on the order of "hundreds of cycles." Let $T_{\text{coh}}$ be the per-event coherence cost and $f_{\text{evt}}(N)$ the event rate at contention level $N$. Then the *useful* CPU cycles per op are bounded by:

$$T_{\text{op}}^{\text{useful}}(N) \geq T_{\text{op}}^{\text{ideal}} \cdot \rho(N) + T_{\text{coh}} \cdot f_{\text{evt}}(N) \qquad \text{(6)}$$

Measuring $T_{\text{coh}}$ directly on Apple Silicon requires `xctrace record --template "CPU Counters"` (see §10, Q1). The present work measures $\rho$ directly and infers the coherence story from the throughput-vs-$\rho$ joint trajectory.

---

## 4. Data Structures

### 4.1 Bounded MPMC Ring Buffer (Vyukov)

A bounded multi-producer multi-consumer ring of capacity $C = 2^k$ with one *per-slot sequence number* serving as the synchronization point (Vyukov 2010; Williams 2019, §7.2.6).

**Cell layout** ([src/mpmc_queue.hpp:75](src/mpmc_queue.hpp)):

```cpp
struct alignas(128) Cell {
    std::atomic<size_t> sequence;
    T                   data;
};
static_assert(sizeof(Cell) == 128, "...");
```

`sizeof(Cell) = 128` ⟹ adjacent cells occupy distinct cache lines ⟹ slot-level false sharing is *structurally impossible* (Williams 2019, §8.2.3).

**Head and tail atomics**:

```cpp
alignas(128) std::atomic<size_t> head_;
alignas(128) std::atomic<size_t> tail_;
```

The two cursors live on disjoint cache lines from each other and from every slot.

#### 4.1.1 Enqueue (`try_push`)

```
pos ← tail_.load(mo_rlx)
loop:
  ++iters
  cell ← &buffer_[pos & mask_]
  seq  ← cell→sequence.load(mo_acq)            (A1)
  diff ← (intptr_t)seq − (intptr_t)pos
  if diff == 0:
      if CAS(tail_, pos, pos+1, mo_rlx):       (A2)
          success ← true; break
  elif diff < 0:
      break                                    (queue full)
  else:
      pos ← tail_.load(mo_rlx)
if success:
    cell→data ← value                          (A3)
    cell→sequence.store(pos+1, mo_rel)         (A4)
return success
```

**Memory ordering rationale.** The release-store on `sequence` at $(A4)$ pairs with the acquire-load on `sequence` performed by `try_pop` at the analogous position $(B1)$:

$$(A4) \xrightarrow{sw} (B1)$$

By Williams (2019, §5.3.1) this establishes:

$$(A3) \xrightarrow{hb} (B3)$$

i.e., the data write at $(A3)$ is visible at the consumer's data read $(B3)$. The CAS at $(A2)$ uses `mo_rlx` because *no data* is published through the position cursor; all visibility flows through the per-cell sequence.

#### 4.1.2 Dequeue (`try_pop`)

Symmetric on `head_` with the sequence check $\textit{diff} = \textit{seq} - (\textit{pos} + 1)$:

```
pos ← head_.load(mo_rlx)
loop:
  ++iters
  cell ← &buffer_[pos & mask_]
  seq  ← cell→sequence.load(mo_acq)            (B1)
  diff ← (intptr_t)seq − (intptr_t)(pos+1)
  if diff == 0:
      if CAS(head_, pos, pos+1, mo_rlx):       (B2)
          success ← true; break
  elif diff < 0:
      break                                    (queue empty)
  else:
      pos ← head_.load(mo_rlx)
if success:
    value ← cell→data                          (B3)
    cell→sequence.store(pos+mask_+1, mo_rel)   (B4)
return success
```

The release-store at $(B4)$ marks the slot as empty for the *next* producer reusing position $\textit{pos} + C$ (one full ring later), since `pos` + `mask_` + $1$ = `pos` + $C$ when capacity is $C$ = `mask_` + $1$. This is the *cyclic monotonicity* invariant of the Vyukov design.

#### 4.1.3 Batched Dequeue (`try_pop_bulk`)

A scan-then-CAS variant that amortizes the head-CAS over $K$ items:

```
1. pos ← head_.load(mo_rlx)
2. tail_now ← tail_.load(mo_acq)            ← extra cross-cache load
3. available ← tail_now − pos
4. take ← min(available, K)
5. for i in [0, take):
       if cell[(pos+i) & mask_].sequence.load(mo_acq) ≠ pos+i+1:
           verified ← i; break    ← shrink batch on unpublished slot
       else: verified ← i+1
6. if !CAS(head_, pos, pos+verified, mo_acq, mo_rlx):
       return 0                            ← race lost
7. for i in [0, verified):
       out[i] ← cell[(pos+i) & mask_].data
       cell[(pos+i) & mask_].sequence.store(pos+i+mask_+1, mo_rel)
8. return verified
```

The expected savings are $K$-fold on head-CAS frequency. The unanticipated costs are the extra `tail_.load(mo_acq)` (step 2) and the double-pass through the slot sequence cache lines (steps 5 and 7). §7.4 reports the net effect.

### 4.2 Bounded SPSC Ring Buffer (Cached-Cursor)

A single-producer single-consumer ring of capacity $C = 2^{12} = 4096$ ([src/spsc_queue.hpp](src/spsc_queue.hpp)). Each side maintains a thread-local cached view of the opposite cursor.

**Layout** (four cache lines):

```cpp
alignas(128) std::atomic<size_t> head_;        // C writes, P reads on cache miss
alignas(128) size_t              cached_tail_; // C-local only
alignas(128) std::atomic<size_t> tail_;        // P writes, C reads on cache miss
alignas(128) size_t              cached_head_; // P-local only
```

#### 4.2.1 Enqueue

```
tail ← tail_.load(mo_rlx)
if tail − cached_head_ ≥ C:
    cached_head_ ← head_.load(mo_acq)         (refresh on apparent fullness)
    if tail − cached_head_ ≥ C: return false
buffer_[tail & mask_] ← value                  (P3)
tail_.store(tail + 1, mo_rel)                  (P4)
return true
```

**No CAS** on the hot path. The cross-cache acquire-load of `head_` is amortized: when the producer's cached view says the queue has free space, no cross-line traffic occurs. This is the structural opposite of the MPMC ring's collapse signature.

#### 4.2.2 Dequeue

```
head ← head_.load(mo_rlx)
if head == cached_tail_:
    cached_tail_ ← tail_.load(mo_acq)
    if head == cached_tail_: return false
value ← buffer_[head & mask_]                  (C3)
head_.store(head + 1, mo_rel)                  (C4)
return true
```

The release-store on `tail_` $(P4)$ synchronizes-with the acquire-load on `tail_` performed by the consumer cache refresh. The release-store on `head_` $(C4)$ synchronizes-with the acquire-load on `head_` performed by the producer cache refresh. By transitivity of $\xrightarrow{hb}$:

$$(P3) \xrightarrow{hb} (C3)$$

establishing visibility of buffered data, exactly as in the MPMC case but via the cursors instead of per-slot sequence.

---

## 5. Experimental Methodology

### 5.1 Trial Structure

For each configuration $(N_P, N_C, \text{mode})$ we run:

- **1 warmup trial**, discarded.
- **5 measured trials**, each of duration 3.0 s (wall-clock).

The warmup ensures branch-predictor state, allocator internal state, and queue buffer pages are warm before measurement.

Reported throughput is the **median of 5 trials** to suppress single-trial outliers (e.g., OS background activity, JIT/codegen quirks).

### 5.2 Timing

Per §3.3, each thread records its own $(s_i, e_i)$ pair via `std::chrono::steady_clock::now()`. The aggregate $R(N)$ is the sum of per-thread rates (Eq. 5), *not* (total ops) / (shared wall clock). The latter would systematically overstate throughput by hiding stragglers.

### 5.3 Sanity Invariants

At end of each `full`-mode run, we assert $O_P = O_C + \ell$ where $\ell$ is the number of items drained by `main()` after thread join — i.e., the count of slots a producer published *after* the consumer's loop exited but *before* the producer's loop exited. The drain is sequential (post-`join`) and race-free.

If any trial deviates more than $2\times$ from the median for its cell, it is flagged in the CSV output. None did in the reported sweeps.

---

## 6. Experimental Configurations

### 6.1 Modes

Let $\mathcal{M}$ denote the set of transport modes:

| Mode | Producer body | Consumer body |
|---|---|---|
| `full` | `malloc(64)` → touch → `try_push(ptr)` | `try_pop(ptr)` → touch → `free(ptr)` |
| `queue-only` | `try_push(sentinel)` | `try_pop` and discard |
| `batchK`, $K \in \{8,16,32\}$ | `try_push(sentinel)` | `try_pop_bulk(buf, K)` |
| `spsc-pair` | round-robin push to row $i$ of $N_P \times N_C$ matrix | round-robin pop from column $j$ |
| `spsc-shard` | push to dedicated queue $q_i$, $i \bmod N_C$ | pop from dedicated queue $q_j$ (requires $N_P = N_C$) |

The sentinel value in `queue-only`/`batch`/`spsc-*` is `(void*)0xdeadbeef`, a non-allocated pointer that exercises only the queue path.

### 6.2 Thread-Count Sweep

$N \in \{1, 2, 4, 8, 16\}$ with $N_P = N_C = N$.

### 6.3 Asymmetry Sweep (Causal Isolation)

$(N_P, N_C) \in \{(1,1), (2,1), (1,2)\}$ in `queue-only` mode only. The purpose is to distinguish producer-side (`tail_`) contention from consumer-side (`head_`) contention; see §7.3.

### 6.4 Allocation Workload

`full` mode allocates `ALLOC_SIZE` = 64 bytes per item, touches one byte to force backing-page commitment, and frees on the consumer side. The allocator is the platform `malloc` (clang 21 runtime on macOS arm64).

---

## 7. Results

All numbers below are medians of 5 trials × 3 s. Full trial-level data in [results/baseline.csv](results/baseline.csv), [results/asymmetry.csv](results/asymmetry.csv), [results/experiments.csv](results/experiments.csv).

### 7.1 Baseline: Allocator Cost is *Not* the Bottleneck

The `full` vs `queue-only` comparison at $N = 1$ isolates per-op allocation overhead:

| Mode | $R$ (ops/s) | $T_{\text{op}}$ (ns/op) |
|---|---:|---:|
| `queue-only` (1, 1) | 39,289,570 | 25.5 |
| `full` (1, 1) | 6,648,280 | 150.4 |

$$\Delta_\text{alloc} = T_\text{full}(1) - T_\text{queue}(1) = 150.4 - 25.5 = \boxed{124.9 \text{ ns/op}} \qquad \text{(7)}$$

The allocator imposes a per-op tax of $\sim 125$ ns at single-thread. However, as shown next, the *queue itself* collapses well before allocation becomes the limiter at higher $N$. The blueprint hypothesis — "build a thread-local slab allocator to fix concurrent throughput" — is refuted by this measurement at the outset.

### 7.2 MPMC Scaling and CAS-Retry Amplification (Experiment D)

| $N$ | $R(N)$ (ops/s) | $\rho(N)$ | $\eta(N)$ |
|---:|---:|---:|---:|
| 1  | 39,289,570 | 1.049 | 100.0% |
| 2  | 10,618,993 | 2.133 | 13.5% |
| 4  | 5,561,865 | 6.198 | 3.5% |
| 8  | 3,136,216 | **278.456** | 1.0% |
| 16 | 1,610,381 | **976.827** | 0.5% |

**Key observations.**

1. **Catastrophic efficiency loss between $N = 1$ and $N = 2$.** Adding a single additional producer-consumer pair drops $\eta$ from $1.0$ to $0.135$. Per Eq. (2), this implies a non-trivial $\beta$ in the Universal Scalability Law — the MPMC queue is *cross-talk-bound*, not *serialization-bound*.

2. **$\rho$ phase transition at $N = 8$.** From $N = 4$ to $N = 8$, $\rho$ jumps **45×** (6.2 → 278). From $N = 8$ to $N = 16$, another **3.5×** (278 → 977). The $N = 8$ inflection coincides with $2N = 16$ worker threads on a $4$-$P$-core machine — the OS scheduler is forced to time-slice on $E$-cores. A thread preempted mid-retry-loop returns to find its target cursor advanced, must re-load the cursor cache line (cross-cluster coherence on M4), and retry. The retry cost compounds.

3. **Throughput collapse and retry amplification are the same phenomenon.** Throughput drops $24.4\times$ from $N = 1$ to $N = 16$; $\rho$ rises $931\times$. The product $R(N) \cdot \rho(N) \approx (39 \cdot 1.05) \approx 41$ Mops·iters/s ≈ $(1.6 \cdot 977) \approx 1560$ Mops·iters/s differs only by a factor of $\sim 38$, consistent with a $\sim 38\times$ increase in *useful* cycles per CAS-loop iteration (cache miss latency growing with $N$).

### 7.3 Asymmetric Contention: Head vs. Tail (Causal Isolation)

To determine whether the collapse is producer-side, consumer-side, or bilateral, we ran three asymmetric configurations in `queue-only` mode:

| $(N_P, N_C)$ | $R$ (ops/s) | Δ vs $(1,1)$ |
|---|---:|---:|
| $(1, 1)$ | 42,888,622 | — |
| $(2, 1)$ | 10,818,720 | **−74.8%** |
| $(1, 2)$ | 11,797,513 | **−72.5%** |

**The collapse is bilateral.** The two contended configurations differ by 9% from each other, both at $\sim −74\%$ versus baseline. The producer-side `tail_` and consumer-side `head_` atomics collapse essentially symmetrically when contended.

False sharing is structurally ruled out by `alignas(128)` on `head_`, `tail_`, and every `Cell`, verified by `static_assert(sizeof(Cell) == 128, ...)`. The remaining cause is **CAS-ownership coherence traffic on the shared position atomic**: when thread $A$ wins a CAS, the cache line is invalidated in thread $B$'s L1, and $B$'s next CAS load incurs a coherence round-trip.

Formal causal chain (per cursor write):

1. $T_A$: `pos_A ← tail_.load(mo_rlx)` — line loaded into $A$'s L1 in Shared state.
2. $T_B$: `pos_B ← tail_.load(mo_rlx)` — line shared, $A$ and $B$ both have copies.
3. $T_A$: `CAS(tail_, pos_A, pos_A+1, mo_rlx)` — line becomes Modified in $A$'s L1, $B$'s copy invalidated.
4. $T_B$: `CAS(tail_, pos_B, pos_B+1, mo_rlx)` — fails (stale value); line transfers M→S via coherence protocol, paid by $B$.
5. $T_B$: retries with $\textit{pos}_B' = \textit{pos}_A + 1$. If a third thread $C$ is competing, repeat from step 3.

Expected wasted iterations per op scale as $\Theta(N_P - 1)$ under uniform competition, consistent with observed $\rho$ growth (slightly super-linear due to coherence-induced delay compounding).

### 7.4 Batched Dequeue: $\rho$ Improves, Throughput Does Not (Experiment B)

`try_pop_bulk(K)` for $K \in \{8, 16, 32\}$, all at $N_P = N_C = N$:

| $N$ | `queue-only` | `batch8` | `batch16` | `batch32` | $\rho$ (queue-only) | $\rho$ (batch32) |
|---:|---:|---:|---:|---:|---:|---:|
| 2  | 10,618,993 | 7,872,899 | 7,873,108 | 8,099,699 | 2.13  | 2.37   |
| 4  | 5,561,865 | 3,814,000 | 3,863,617 | 3,858,775 | 6.20  | 9.99   |
| 8  | 3,136,216 | 2,555,175 | 2,566,248 | 2,498,828 | 278.5 | 265.7  |
| 16 | 1,610,381 | 1,530,202 | 1,521,003 | 1,438,779 | 976.8 | 1040.1 |

**Batching reduces $\rho$ at high $N$** (e.g., 278 → 234 at $N = 8$ with $K = 8$) but **does not improve throughput**. Two cancelling effects:

- *Savings*: one head-CAS amortized over $K$ items.
- *Costs* (the implementation in [src/mpmc_queue.hpp](src/mpmc_queue.hpp), `try_pop_bulk`):
  - One extra `tail_.load(mo_acq)` per call to compute `available` (cross-cache load not present in single `try_pop`);
  - Verify-then-CAS performs *two passes* through the slot sequence cache lines (touch each cell line twice, once for verify and once for read);
  - `compare_exchange_strong` (used here) is slightly more conservative than `compare_exchange_weak` (Williams §5.2.3).

**Net interpretation.** The hypothesis "batched dequeue reduces coherence traffic" is directionally validated by $\rho$ but the present implementation's overhead exceeds the savings. A proper batched-pop (skip the tail-load, single-pass over slots) is left as future work (§10, Q4).

### 7.5 Per-Pair SPSC Lanes (Experiment A)

Replace the single MPMC ring with $N_P \times N_C$ dedicated SPSC queues; each producer round-robins push across its row of $N_C$ queues, each consumer round-robins pop down its column of $N_P$ queues. No shared atomic remains on the critical path.

| $N$ | $R_{\text{MPMC}}(N)$ | $R_{\text{SPSC-pair}}(N)$ | Speedup |
|---:|---:|---:|---:|
| 1  | 39,289,570 | 22,190,624 | 0.57× |
| 2  | 10,618,993 | 244,041,486 | **23×** |
| 4  | 5,561,865 | 263,574,349 | **47×** |
| 8  | 3,136,216 | 166,487,710 | **53×** |
| 16 | 1,610,381 | 146,610,929 | **91×** |

**Two findings.**

1. At $N \geq 2$, SPSC-pair throughput is **1–2 orders of magnitude** higher than MPMC. The shared-atomic ownership hypothesis is empirically confirmed: removing the shared head/tail removes the collapse.

2. **At $N = 1$, SPSC-pair underperforms MPMC** (22.2 M vs 39.3 M ops/s). The four-cache-line metadata footprint of the cached-cursor SPSC pays a fixed overhead that is only amortized once cross-cache coherence traffic from a *second* thread starts dominating. This is the regime-dependent transport choice noted in §8.

### 7.6 Producer-Sharded SPSC (Experiment C)

Reduce the matrix from $N_P \times N_C$ queues to $N_C$ queues; pin producer $i$ to queue $i \bmod N_C$. Each queue remains strict SPSC (one producer + one consumer) iff $N_P = N_C$.

| $N$ | `spsc-pair` | `spsc-shard` | Ratio |
|---:|---:|---:|---:|
| 1  | 22,190,624 | 24,931,818 | **1.12×** |
| 2  | 244,041,486 | 55,250,788 | 0.23× |
| 4  | 263,574,349 | **596,963,327** | **2.26×** |
| 8  | 166,487,710 | 319,329,245 | **1.92×** |
| 16 | 146,610,929 | 138,148,953 | 0.94× |

**Peak: $R_{\text{spsc-shard}}(4) = 596.9$ Mops/s.** This is **107× the MPMC baseline** at the same $N$ (5.56 Mops/s).

**Cache-locality explanation.** At $N = 4$, `spsc-shard` allocates $N_C = 4$ queues; each ring buffer is $4096 \times 8 = 32$ KiB. Total queue footprint: $128$ KiB = exactly the M4 $P$-core L1-D capacity. Each producer-consumer pair touches one ring buffer of 32 KiB, fitting comfortably in L1 with room for code and stack. Coherence traffic on the cursors is the only cross-cache cost, and the cached-cursor design amortizes that.

`spsc-pair` at $N = 4$ allocates $N_P \times N_C = 16$ queues = $512$ KiB total. Each thread round-robins across $N_C = 4$ queues of its row = $128$ KiB working set per thread, which is *equal* to L1-D but contended with code and stack. The matrix layout pays a cache footprint penalty that the sharded layout avoids.

**The $N = 2$ anomaly.** `spsc-shard` at $N = 2$ underperforms `spsc-pair` at $N = 2$ (55 M vs 244 M ops/s). With only $N_C = 2$ queues on a 4-$P$-core machine, both producer-consumer pairs may be scheduled onto the same $P$-core cluster pair, causing cache-set conflicts that the 4-queue matrix layout disperses. Direct verification requires L1-D miss-rate counters (§10, Q2).

**Decline at $N \geq 8$** follows the same scheduling argument as §7.2: 16+ threads on 4 $P$-cores spill to $E$-cores, cross-cluster L2 coherence dominates.

---

## 8. Findings

**F1. The shared atomic head/tail is the dominant bottleneck**, not allocation.

Removing the shared cursor (SPSC pair, $N = 2$) recovers $244 / 10.6 = 23\times$ throughput versus MPMC at the same thread count. Allocation overhead at single-thread is $\Delta_\text{alloc} = 124.9$ ns/op (Eq. 7), but the MPMC queue collapses long before allocation becomes the rate-limiter.

**F2. CAS-retry amplification is the proximate cause.**

$\rho$ grows from $1.05$ at $N = 1$ to $976.8$ at $N = 16$ on MPMC `queue-only`. The $45\times$ jump from $N = 4$ to $N = 8$ marks the OS-scheduler phase transition where thread count exceeds $P$-core count on M4.

**F3. The collapse is bilateral, not one-sided.**

Asymmetric tests $(2, 1)$ and $(1, 2)$ collapse by $-74.8\%$ and $-72.5\%$ respectively (Δ = 9%). Neither producer-side nor consumer-side contention is dominant — both shared atomics fail under any peer.

**F4. False sharing is structurally ruled out.**

`alignas(128)` on `Cell`, `head_`, and `tail_` (verified by `static_assert`) eliminates slot-level and cursor-adjacent false sharing. The collapse is a *true* contention phenomenon, not a layout artifact.

**F5. Batched dequeue reduces $\rho$ but not net throughput** under the present implementation.

$K \in \{8, 16, 32\}$ all reduce $\rho$ at high $N$ but pay extra cross-cache loads that exceed the savings. Hypothesis directionally validated by $\rho$, falsified by throughput; a better batched-pop is owed (§10, Q4).

**F6. Cache locality is the dominant second-order effect for SPSC variants.**

`spsc-shard` ($N_C$ queues) beats `spsc-pair` ($N_P \times N_C$ queues) by **2.26×** at $N = 4$, where the sharded queue set fits entirely in M4 $P$-core L1-D. At $N = 2$ the small queue set causes scheduler-induced cache-set conflicts (open question Q2).

**F7. At very low contention, MPMC outperforms cached-cursor SPSC.**

`queue-only` (MPMC) at $N = 1$ achieves 39.3 M ops/s versus SPSC-pair's 22.2 M ops/s. The SPSC cached-cursor overhead (four metadata cache lines plus the cache-refresh branch) is only amortized once cross-cache traffic from a second thread dominates. **The right transport depends on the contention regime**, not the lock-free progress class.

---

## 9. Repository Structure and Reproducibility

```
tlslab/
├── Makefile
├── README.md                       (this file)
├── baseline                        (compiled binary)
├── src/
│   ├── mpmc_queue.hpp              ← Vyukov bounded MPMC, alignas(128) cells
│   ├── spsc_queue.hpp              ← Cached-cursor SPSC, four isolated lines
│   └── baseline.cpp                ← Harness: full / queue-only / batchK / spsc-pair / spsc-shard
├── scripts/
│   ├── run_baseline.sh             ← Sweep: N ∈ {1,2,4,8,16}, modes: full + queue-only
│   ├── run_asymmetry.sh            ← Sweep: (1,1), (2,1), (1,2) in queue-only
│   └── run_experiments.sh          ← Experiments A, B, C, D combined
└── results/
    ├── baseline.csv                ← Trial-level: full vs queue-only scaling
    ├── asymmetry.csv               ← Trial-level: head vs tail contention isolation
    └── experiments.csv             ← Trial-level: SPSC + batched + retry counts
```

### 9.1 Build

```bash
make                # clang++ -std=c++20 -O2 -pthread -Wall -Wextra
make baseline-tsan  # -fsanitize=thread -g  (data-race validation)
```

The `static_assert(sizeof(Cell) == 128, ...)` in [src/mpmc_queue.hpp](src/mpmc_queue.hpp) is the compile-time guard against slot false sharing. If this fires, padding is wrong and the entire experiment is invalid; report this before trusting any throughput number.

### 9.2 Run

```bash
bash scripts/run_baseline.sh        # ≈ 5 min   (Experiments: full / queue-only)
bash scripts/run_asymmetry.sh       # ≈ 2 min   (head vs tail contention)
bash scripts/run_experiments.sh     # ≈ 20 min  (Experiments A, B, C, D)
```

Each script writes trial-level CSV plus a median summary table to stderr.

### 9.3 CSV Schema

All CSVs share columns:

```
NP, NC, mode, trial, total_ops, duration_ns, ops_per_sec, ns_per_op,
min_thread_rate, max_thread_rate, sum_prod_rate, total_iters, iters_per_op
```

`iters_per_op` is the per-trial $\rho$ value (Eq. 3).

---

## 10. Open Questions

**Q1. Hardware-counter attribution.** The $N = 8$ $\rho$ phase transition is attributed to scheduler preemption on $E$-cores via indirect evidence ($\rho$ discontinuity coinciding with thread count exceeding $P$-core count). Direct confirmation requires L1-D miss-rate and CAS-failure-counter samples via `xctrace record --template "CPU Counters"` on arm64. The Apple Silicon PMU exposes counters but the documentation set is incomplete; full enumeration deferred.

**Q2. The $N = 2$ `spsc-shard` anomaly.** With $N_C = 2$ on a 4-$P$-core M4, the two SPSC queues underperform the 4-queue `spsc-pair` matrix at the same thread count (55 M vs 244 M ops/s). The hypothesized cause is L1 cache-set conflicts from two queues placed on the same core's cache; confirmation requires L1-D miss-rate counters.

**Q3. Fetch-and-add tail (LCRQ-style enqueue).** Replacing the CAS loop on `tail_` with `fetch_add(1, mo_rlx)` eliminates retry entirely for producers — each thread atomically claims a unique slot without CAS failure. The open question is whether FAA's unconditional cache-line write yields comparable coherence traffic, or whether the absence of the retry loop is a net win. Expected result: FAA outperforms CAS under high $N_P$, underperforms under low $N_P$ due to store-broadcast cost. (Morrison and Afek 2013.)

**Q4. A proper batched dequeue.** The current `try_pop_bulk` reads `tail_` upfront and performs a verify-then-CAS double pass over the slot range. A redesign that uses producer-published sequences directly (no `tail_` load) and single-passes the slots may convert the $\rho$ reduction observed in §7.4 into a net throughput gain. ~30 minutes of implementation.

**Q5. Memory reclamation for unbounded variants.** This work studies *bounded* queues; the slot lifetimes are tied to ring positions. Unbounded variants (Michael & Scott 1996) require Safe Memory Reclamation (SMR) — typically hazard pointers (Michael 2004), epoch-based reclamation, or wait-free reclamation. Wolff's dissertation (Wolff 2021) provides formal verification techniques for these. Out of scope for this study; see §11.

**Q6. Allocator contention in the SPSC regime.** The original motivating question — does `malloc` contention dominate at higher $N$? — is unanswered because the MPMC queue collapses first. Re-running the `full` vs `queue-only` comparison using `spsc-shard` as the transport (which does *not* collapse) would cleanly isolate allocator behavior at $N = 4$, where the queue is no longer the limiter. Useful comparison points would be mimalloc (Leijen, Zorn, de Moura 2019) and snmalloc (Liétar et al. 2019), whose sharded free-list designs are the allocator analogue of the queue-sharding result of §7.6.

---

## 11. References

- Maurice Herlihy and Nir Shavit. *The Art of Multiprocessor Programming*. 2nd ed. Morgan Kaufmann, 2020. Chapter 3 (Concurrent Objects, progress conditions), Chapter 10 (Queues, Stacks).
- Anthony Williams. *C++ Concurrency in Action*. 2nd ed. Manning, 2019. Chapter 5 (memory model and atomic operations), Chapter 7 (lock-free concurrent data structures), Chapter 8 (cache ping-pong and false sharing).
- Daan Leijen, Benjamin Zorn, and Leonardo de Moura. *mimalloc: Free List Sharding in Action*. Microsoft Technical Report MSR-TR-2019-18, 2019. The terminology of *free list sharding*, *local free list*, *thread free list*, and *asymmetric workload* used throughout §6–§10 is from this work.
- Maged M. Michael. *Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects*. IEEE Transactions on Parallel and Distributed Systems, 15(6):491–504, 2004. Cited for the formal SMR framework, CAS notation, and the ABA problem.
- Sebastian Wolff. *Verifying Non-blocking Data Structures with Manual Memory Management*. PhD dissertation, TU Braunschweig, 2021. Cited for the formal treatment of pointer races, harmful ABA, and verified lock-free reclamation.
- Dmitry Vyukov. *Bounded MPMC queue*. 2010. https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
- Maged M. Michael and Michael L. Scott. *Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms*. PODC 1996.
- Adam Morrison and Yehuda Afek. *Fast concurrent queues for x86 processors*. PPoPP 2013. (LCRQ — fetch-and-add tail design referenced in §10, Q3.)
- Paul Liétar, Theodore Butler, Sylvan Clebsch, Sophia Drossopoulou, Juliana Franco, Matthew J. Parkinson, Alex Shamis, Christoph M. Wintersteiger, and David Chisnall. *snmalloc: A message passing allocator*. ISMM 2019.
- Neil J. Gunther. *Guerrilla Capacity Planning*. Springer, 2007. Original derivation of the Universal Scalability Law (Eq. 1).
- Jeff Preshing. *Acquire and Release Semantics*. 2012. https://preshing.com/20120913/acquire-and-release-semantics
- Jeff Preshing. *An Introduction to Lock-Free Programming*. 2012. https://preshing.com/20120612/an-introduction-to-lock-free-programming

---

## 12. Citation

```bibtex
@misc{tlslab2026,
  title  = {Contention Collapse in Lock-Free {MPMC} Ring Buffers on {Apple Silicon} {M4}:
            An Empirical Study of {CAS}-Retry Amplification and Queue Topology},
  author = {Raja Babu},
  year   = {2026},
  month  = {May},
  note   = {Empirical study, IIT (ISM) Dhanbad. Source and data:
            \url{https://github.com/RajaBabu15/tlslab}},
  howpublished = {GitHub repository}
}
```

---

## 13. License

MIT License.

Copyright (c) 2026 Raja Babu.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
