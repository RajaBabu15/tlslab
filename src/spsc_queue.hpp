#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>

template <typename T>
class SPSCQueue {
public:
    explicit SPSCQueue(size_t capacity)
        : capacity_(capacity), mask_(capacity - 1),
          buffer_(new T[capacity]) {
        assert((capacity & (capacity - 1)) == 0 && "capacity must be a power of 2");
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        cached_head_ = 0;
        cached_tail_ = 0;
    }

    ~SPSCQueue() { delete[] buffer_; }

    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    bool try_push(const T& value) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail - cached_head_ >= capacity_) {
            cached_head_ = head_.load(std::memory_order_acquire);
            if (tail - cached_head_ >= capacity_) return false;
        }
        buffer_[tail & mask_] = value;
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    bool try_pop(T& value) {
        const size_t head = head_.load(std::memory_order_relaxed);
        if (head == cached_tail_) {
            cached_tail_ = tail_.load(std::memory_order_acquire);
            if (head == cached_tail_) return false;
        }
        value = buffer_[head & mask_];
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

private:
    const size_t capacity_;
    const size_t mask_;
    T* buffer_;

    // Four isolated cache lines. cached_tail_ lives with head_ on the
    // consumer side; cached_head_ lives with tail_ on the producer side.
    // Reading them off-thread is intentional: that's the cache-miss event
    // the cached-cursor pattern is amortizing.
    alignas(128) std::atomic<size_t> head_;
    alignas(128) size_t cached_tail_;
    alignas(128) std::atomic<size_t> tail_;
    alignas(128) size_t cached_head_;
};
