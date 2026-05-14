#include "mpmc_queue.hpp"
#include "spsc_queue.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <pthread.h>
#include <string>
#include <sys/qos.h>
#include <thread>
#include <vector>

constexpr size_t MPMC_CAP = 1u << 14;
constexpr size_t SPSC_CAP = 1u << 12;
constexpr size_t ALLOC_SIZE = 64;
constexpr uintptr_t SENTINEL = 0xdeadbeefULL;

struct alignas(128) ThreadStats {
    std::atomic<uint64_t> ops{0};
    std::atomic<uint64_t> queue_iters{0};
    int64_t start_ns{0};
    int64_t end_ns{0};
};

static_assert(sizeof(ThreadStats) == 128,
              "ThreadStats must occupy exactly one cache line");

static std::atomic<bool> g_stop{false};

static inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static inline void cpu_yield() {
    asm volatile("yield" ::: "memory");
}

static void set_qos_hint() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

static void producer_full(MPMCQueue<void*>* q, ThreadStats* s) {
    set_qos_hint();
    s->start_ns = now_ns();
    uint64_t count = 0;
    uint64_t iters = 0;
    while (!g_stop.load(std::memory_order_relaxed)) {
        void* p = std::malloc(ALLOC_SIZE);
        if (!p) std::abort();
        *static_cast<volatile char*>(p) = static_cast<char>(count);
        while (!q->try_push(p, &iters)) {
            if (g_stop.load(std::memory_order_relaxed)) {
                std::free(p);
                goto done;
            }
            cpu_yield();
        }
        ++count;
    }
done:
    s->end_ns = now_ns();
    s->queue_iters.store(iters, std::memory_order_release);
    s->ops.store(count, std::memory_order_release);
}

static void consumer_full(MPMCQueue<void*>* q, ThreadStats* s) {
    set_qos_hint();
    s->start_ns = now_ns();
    uint64_t count = 0;
    uint64_t iters = 0;
    void* p = nullptr;
    while (!g_stop.load(std::memory_order_relaxed)) {
        if (q->try_pop(p, &iters)) {
            volatile char c = *static_cast<char*>(p);
            (void)c;
            std::free(p);
            ++count;
        } else {
            cpu_yield();
        }
    }
    s->end_ns = now_ns();
    s->queue_iters.store(iters, std::memory_order_release);
    s->ops.store(count, std::memory_order_release);
}

static void producer_qonly(MPMCQueue<void*>* q, ThreadStats* s) {
    set_qos_hint();
    s->start_ns = now_ns();
    uint64_t count = 0;
    uint64_t iters = 0;
    void* sentinel = reinterpret_cast<void*>(SENTINEL);
    while (!g_stop.load(std::memory_order_relaxed)) {
        while (!q->try_push(sentinel, &iters)) {
            if (g_stop.load(std::memory_order_relaxed)) goto done;
            cpu_yield();
        }
        ++count;
    }
done:
    s->end_ns = now_ns();
    s->queue_iters.store(iters, std::memory_order_release);
    s->ops.store(count, std::memory_order_release);
}

static void consumer_qonly(MPMCQueue<void*>* q, ThreadStats* s) {
    set_qos_hint();
    s->start_ns = now_ns();
    uint64_t count = 0;
    uint64_t iters = 0;
    void* p = nullptr;
    while (!g_stop.load(std::memory_order_relaxed)) {
        if (q->try_pop(p, &iters)) {
            ++count;
        } else {
            cpu_yield();
        }
    }
    s->end_ns = now_ns();
    s->queue_iters.store(iters, std::memory_order_release);
    s->ops.store(count, std::memory_order_release);
}

static void consumer_qonly_batch(MPMCQueue<void*>* q, ThreadStats* s, size_t batch) {
    set_qos_hint();
    s->start_ns = now_ns();
    uint64_t count = 0;
    uint64_t iters = 0;
    constexpr size_t MAX_BATCH = 64;
    void* buf[MAX_BATCH];
    if (batch > MAX_BATCH) batch = MAX_BATCH;
    while (!g_stop.load(std::memory_order_relaxed)) {
        size_t got = q->try_pop_bulk(buf, batch, &iters);
        if (got > 0) {
            count += got;
        } else {
            cpu_yield();
        }
    }
    s->end_ns = now_ns();
    s->queue_iters.store(iters, std::memory_order_release);
    s->ops.store(count, std::memory_order_release);
}

static void producer_spsc_pair(SPSCQueue<void*>** queues, int prod_idx, int NC,
                               ThreadStats* s) {
    set_qos_hint();
    s->start_ns = now_ns();
    uint64_t count = 0;
    void* sentinel = reinterpret_cast<void*>(SENTINEL);
    int j = 0;
    SPSCQueue<void*>** mine = &queues[prod_idx * NC];
    while (!g_stop.load(std::memory_order_relaxed)) {
        if (mine[j]->try_push(sentinel)) ++count;
        else cpu_yield();
        j = (j + 1 == NC) ? 0 : j + 1;
    }
    s->end_ns = now_ns();
    s->ops.store(count, std::memory_order_release);
}

static void consumer_spsc_pair(SPSCQueue<void*>** queues, int cons_idx,
                               int NP, int NC, ThreadStats* s) {
    set_qos_hint();
    s->start_ns = now_ns();
    uint64_t count = 0;
    void* p = nullptr;
    int i = 0;
    while (!g_stop.load(std::memory_order_relaxed)) {
        if (queues[i * NC + cons_idx]->try_pop(p)) ++count;
        else cpu_yield();
        i = (i + 1 == NP) ? 0 : i + 1;
    }
    s->end_ns = now_ns();
    s->ops.store(count, std::memory_order_release);
}

static void producer_spsc_shard(SPSCQueue<void*>** queues, int prod_idx,
                                ThreadStats* s) {
    set_qos_hint();
    s->start_ns = now_ns();
    uint64_t count = 0;
    void* sentinel = reinterpret_cast<void*>(SENTINEL);
    SPSCQueue<void*>* q = queues[prod_idx];
    while (!g_stop.load(std::memory_order_relaxed)) {
        while (!q->try_push(sentinel)) {
            if (g_stop.load(std::memory_order_relaxed)) goto done;
            cpu_yield();
        }
        ++count;
    }
done:
    s->end_ns = now_ns();
    s->ops.store(count, std::memory_order_release);
}

static void consumer_spsc_shard(SPSCQueue<void*>** queues, int cons_idx,
                                ThreadStats* s) {
    set_qos_hint();
    s->start_ns = now_ns();
    uint64_t count = 0;
    void* p = nullptr;
    SPSCQueue<void*>* q = queues[cons_idx];
    while (!g_stop.load(std::memory_order_relaxed)) {
        if (q->try_pop(p)) ++count;
        else cpu_yield();
    }
    s->end_ns = now_ns();
    s->ops.store(count, std::memory_order_release);
}

static int parse_batch_size(const std::string& mode) {
    if (mode.rfind("batch", 0) != 0) return -1;
    return std::atoi(mode.c_str() + 5);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "Usage: %s <NP> <NC> <mode> <duration_sec>\n"
                     "  mode in {full, queue-only, batch<N>, spsc-pair, spsc-shard}\n",
                     argv[0]);
        return 1;
    }
    int NP = std::atoi(argv[1]);
    int NC = std::atoi(argv[2]);
    std::string mode = argv[3];
    double duration_sec = std::atof(argv[4]);
    int batch_size = parse_batch_size(mode);
    bool is_batch_mode = (batch_size > 0);
    if (NP <= 0 || NC <= 0 || duration_sec <= 0.0) {
        std::fprintf(stderr, "bad args\n");
        return 1;
    }
    if (!(mode == "full" || mode == "queue-only" || mode == "spsc-pair" ||
          mode == "spsc-shard" || is_batch_mode)) {
        std::fprintf(stderr, "bad mode: %s\n", mode.c_str());
        return 1;
    }
    if (mode == "spsc-shard" && NP != NC) {
        std::fprintf(stderr, "spsc-shard requires NP == NC\n");
        return 1;
    }

    std::vector<ThreadStats> prod_stats(NP);
    std::vector<ThreadStats> cons_stats(NC);
    std::vector<std::thread> threads;
    threads.reserve(NP + NC);

    std::unique_ptr<MPMCQueue<void*>> mpmc;
    std::vector<std::unique_ptr<SPSCQueue<void*>>> spsc_owners;
    std::vector<SPSCQueue<void*>*> spsc_ptrs;

    g_stop.store(false, std::memory_order_relaxed);

    if (mode == "full" || mode == "queue-only" || is_batch_mode) {
        mpmc = std::make_unique<MPMCQueue<void*>>(MPMC_CAP);
        for (int i = 0; i < NP; ++i) {
            if (mode == "full") {
                threads.emplace_back(producer_full, mpmc.get(), &prod_stats[i]);
            } else {
                threads.emplace_back(producer_qonly, mpmc.get(), &prod_stats[i]);
            }
        }
        for (int i = 0; i < NC; ++i) {
            if (mode == "full") {
                threads.emplace_back(consumer_full, mpmc.get(), &cons_stats[i]);
            } else if (is_batch_mode) {
                threads.emplace_back(consumer_qonly_batch, mpmc.get(),
                                     &cons_stats[i], static_cast<size_t>(batch_size));
            } else {
                threads.emplace_back(consumer_qonly, mpmc.get(), &cons_stats[i]);
            }
        }
    } else if (mode == "spsc-pair") {
        int nqueues = NP * NC;
        spsc_owners.reserve(nqueues);
        spsc_ptrs.reserve(nqueues);
        for (int k = 0; k < nqueues; ++k) {
            spsc_owners.emplace_back(std::make_unique<SPSCQueue<void*>>(SPSC_CAP));
            spsc_ptrs.push_back(spsc_owners.back().get());
        }
        for (int i = 0; i < NP; ++i) {
            threads.emplace_back(producer_spsc_pair, spsc_ptrs.data(), i, NC,
                                 &prod_stats[i]);
        }
        for (int j = 0; j < NC; ++j) {
            threads.emplace_back(consumer_spsc_pair, spsc_ptrs.data(), j, NP, NC,
                                 &cons_stats[j]);
        }
    } else if (mode == "spsc-shard") {
        spsc_owners.reserve(NP);
        spsc_ptrs.reserve(NP);
        for (int k = 0; k < NP; ++k) {
            spsc_owners.emplace_back(std::make_unique<SPSCQueue<void*>>(SPSC_CAP));
            spsc_ptrs.push_back(spsc_owners.back().get());
        }
        for (int i = 0; i < NP; ++i) {
            threads.emplace_back(producer_spsc_shard, spsc_ptrs.data(), i,
                                 &prod_stats[i]);
        }
        for (int j = 0; j < NC; ++j) {
            threads.emplace_back(consumer_spsc_shard, spsc_ptrs.data(), j,
                                 &cons_stats[j]);
        }
    }

    std::this_thread::sleep_for(std::chrono::nanoseconds(
        static_cast<int64_t>(duration_sec * 1e9)));
    g_stop.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();

    // After join, producers may have published slots that the consumer loops
    // missed (publish race on stop). Drain sequentially here.
    uint64_t leftover = 0;
    if (mpmc) {
        void* p = nullptr;
        while (mpmc->try_pop(p)) {
            if (mode == "full" && reinterpret_cast<uintptr_t>(p) != SENTINEL) {
                std::free(p);
            }
            ++leftover;
        }
    } else {
        for (auto& q : spsc_owners) {
            void* p = nullptr;
            while (q->try_pop(p)) ++leftover;
        }
    }

    uint64_t total_prod_ops = 0, total_cons_ops = 0;
    uint64_t total_iters = 0;
    double sum_prod_rate = 0.0, sum_cons_rate = 0.0;
    double min_rate = std::numeric_limits<double>::infinity();
    double max_rate = 0.0;
    int64_t min_start = std::numeric_limits<int64_t>::max();
    int64_t max_end = 0;

    auto fold = [&](const ThreadStats& s, double& sum_rate) {
        uint64_t ops = s.ops.load(std::memory_order_acquire);
        uint64_t iters = s.queue_iters.load(std::memory_order_acquire);
        total_iters += iters;
        double dur_sec = (s.end_ns - s.start_ns) / 1e9;
        double rate = (dur_sec > 0) ? ops / dur_sec : 0.0;
        sum_rate += rate;
        if (rate < min_rate) min_rate = rate;
        if (rate > max_rate) max_rate = rate;
        if (s.start_ns < min_start) min_start = s.start_ns;
        if (s.end_ns > max_end) max_end = s.end_ns;
        return ops;
    };

    for (int i = 0; i < NP; ++i) total_prod_ops += fold(prod_stats[i], sum_prod_rate);
    for (int i = 0; i < NC; ++i) total_cons_ops += fold(cons_stats[i], sum_cons_rate);

    if (mode == "full" && total_prod_ops != total_cons_ops + leftover) {
        std::fprintf(stderr,
                     "WARN: malloc=%llu free=%llu+leftover=%llu — mismatch %lld\n",
                     static_cast<unsigned long long>(total_prod_ops),
                     static_cast<unsigned long long>(total_cons_ops),
                     static_cast<unsigned long long>(leftover),
                     static_cast<long long>(total_prod_ops) -
                         static_cast<long long>(total_cons_ops + leftover));
        return 2;
    }

    double ops_per_sec = sum_cons_rate;
    double total_ops = static_cast<double>(total_cons_ops);
    double duration_ns = static_cast<double>(max_end - min_start);
    double ns_per_op = (ops_per_sec > 0) ? 1e9 / ops_per_sec : 0.0;
    double total_attempts = static_cast<double>(total_prod_ops + total_cons_ops);
    double iters_per_op = (total_attempts > 0)
                              ? static_cast<double>(total_iters) / total_attempts
                              : 0.0;

    std::printf("%d,%d,%s,%.0f,%.0f,%.2f,%.2f,%.2f,%.2f,%.2f,%llu,%.3f\n",
                NP, NC, mode.c_str(), total_ops, duration_ns,
                ops_per_sec, ns_per_op, min_rate, max_rate, sum_prod_rate,
                static_cast<unsigned long long>(total_iters), iters_per_op);
    return 0;
}
