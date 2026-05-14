#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

template <typename T>
class MPMCQueue {
public:
    explicit MPMCQueue(size_t capacity)
        : capacity_(capacity), mask_(capacity - 1),
          buffer_(new Cell[capacity]) {
        assert((capacity & (capacity - 1)) == 0 && "capacity must be a power of 2");
        for (size_t i = 0; i < capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    ~MPMCQueue() { delete[] buffer_; }

    MPMCQueue(const MPMCQueue&) = delete;
    MPMCQueue& operator=(const MPMCQueue&) = delete;

    bool try_push(const T& value, uint64_t* iter_count_out = nullptr) {
        Cell* cell;
        size_t pos = tail_.load(std::memory_order_relaxed);
        uint64_t iters = 0;
        bool success = false;
        for (;;) {
            ++iters;
            cell = &buffer_[pos & mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed)) {
                    success = true;
                    break;
                }
            } else if (diff < 0) {
                break;
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
        if (success) {
            cell->data = value;
            cell->sequence.store(pos + 1, std::memory_order_release);
        }
        if (iter_count_out) *iter_count_out += iters;
        return success;
    }

    bool try_pop(T& value, uint64_t* iter_count_out = nullptr) {
        Cell* cell;
        size_t pos = head_.load(std::memory_order_relaxed);
        uint64_t iters = 0;
        bool success = false;
        for (;;) {
            ++iters;
            cell = &buffer_[pos & mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed)) {
                    success = true;
                    break;
                }
            } else if (diff < 0) {
                break;
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
        if (success) {
            value = cell->data;
            cell->sequence.store(pos + mask_ + 1, std::memory_order_release);
        }
        if (iter_count_out) *iter_count_out += iters;
        return success;
    }

    size_t try_pop_bulk(T* out, size_t want,
                        uint64_t* iter_count_out = nullptr) {
        if (want == 0) return 0;
        uint64_t iters = 1;
        size_t head_pos = head_.load(std::memory_order_relaxed);
        size_t tail_pos = tail_.load(std::memory_order_acquire);
        size_t available = tail_pos - head_pos;
        if (available == 0) {
            if (iter_count_out) *iter_count_out += iters;
            return 0;
        }
        size_t take = available < want ? available : want;
        size_t verified = 0;
        for (size_t i = 0; i < take; ++i) {
            Cell* cell = &buffer_[(head_pos + i) & mask_];
            size_t want_seq = head_pos + i + 1;
            if (cell->sequence.load(std::memory_order_acquire) != want_seq) {
                break;
            }
            ++verified;
        }
        if (verified == 0) {
            if (iter_count_out) *iter_count_out += iters;
            return 0;
        }
        size_t expected = head_pos;
        if (!head_.compare_exchange_strong(expected, head_pos + verified,
                                           std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            if (iter_count_out) *iter_count_out += iters;
            return 0;
        }
        for (size_t i = 0; i < verified; ++i) {
            Cell* cell = &buffer_[(head_pos + i) & mask_];
            out[i] = cell->data;
            cell->sequence.store(head_pos + i + mask_ + 1,
                                 std::memory_order_release);
        }
        if (iter_count_out) *iter_count_out += iters;
        return verified;
    }

private:
    // Cell is one M4 cache line. The static_assert is the structural guard
    // against slot-adjacent false sharing.
    struct alignas(128) Cell {
        std::atomic<size_t> sequence;
        T data;
    };
    static_assert(sizeof(Cell) == 128, "Cell must occupy exactly one cache line");

    const size_t capacity_;
    const size_t mask_;
    Cell* buffer_;

    alignas(128) std::atomic<size_t> head_;
    alignas(128) std::atomic<size_t> tail_;
};
