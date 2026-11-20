// C++ shim: dnnl::threadpool_interop::threadpool_iface → sinteric_tp_* (the
// library's thread pool). Weak defaults keep onednn-sys tests linkable; the
// library provides the strong definitions.
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

#include <cstdint>
#include <functional>

extern "C" {

__attribute__((weak)) int sinteric_tp_num_threads(void) { return 1; }

__attribute__((weak)) int sinteric_tp_in_parallel(void) { return 0; }

__attribute__((weak)) void sinteric_tp_parallel_for_capped(
        int n, void (*cb)(int, int, void *), void *ctx, int max_threads) {
    (void)max_threads;
    if (cb != nullptr && n > 0) {
        for (int i = 0; i < n; ++i) cb(i, n, ctx);
    }
}

__attribute__((weak)) void sinteric_tp_parallel_for(
        int n, void (*cb)(int, int, void *), void *ctx) {
    if (cb != nullptr && n > 0) {
        // A primitive may precompute partitions; serial execution must run all.
        for (int i = 0; i < n; ++i) cb(i, n, ctx);
    }
}

}

namespace {

// `max_threads_ <= 0` is the whole process pool; otherwise oneDNN sees at
// most that many threads and its parallel_for runs on that many pool workers.
class SntThreadpool final : public dnnl::threadpool_interop::threadpool_iface {
public:
    explicit SntThreadpool(int max_threads) : max_threads_(max_threads) {}

    int get_num_threads() const override {
        const int all = sinteric_tp_num_threads();
        return (max_threads_ > 0 && max_threads_ < all) ? max_threads_ : all;
    }

    bool get_in_parallel() const override { return sinteric_tp_in_parallel() != 0; }

    void parallel_for(int n, const std::function<void(int, int)> &fn) override {
        sinteric_tp_parallel_for_capped(
                n,
                [](int ithr, int nthr, void *ctx) {
                    auto *f = static_cast<const std::function<void(int, int)> *>(ctx);
                    (*f)(ithr, nthr);
                },
                const_cast<void *>(static_cast<const void *>(&fn)),
                get_num_threads());
    }

    uint64_t get_flags() const override { return 0; }

    void wait() override {}

private:
    int max_threads_;
};

} // namespace

extern "C" {

void *sinteric_dnnl_threadpool_create(void) { return new SntThreadpool(0); }

void *sinteric_dnnl_threadpool_create_capped(int max_threads) {
    return new SntThreadpool(max_threads);
}

void sinteric_dnnl_threadpool_destroy(void *p) {
    delete static_cast<SntThreadpool *>(p);
}

}
