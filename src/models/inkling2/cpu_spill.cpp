#include "models/inkling2/cpu_spill.hpp"

#include "basalt/core/device.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/log.hpp"
#include "basalt/artifact/dequant.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/platform/compiler.hpp"
#include "models/inkling2/model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

namespace basalt::models::inkling2 {
namespace cpu = kernels::cpu;
namespace {
constexpr int kUsed = 8;
constexpr int kPairs = kDecodeRows * kUsed;
/// [W2] Visit slots: a decode window is 48 visits, so every visit owns one, and the poller can never miss one.
constexpr int kRing = 48;   // [W2] one slot per visit of a decode window: a slot shared by two visits would let the
void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
// [W2] A/B switch for the hand-off.  The doorbell is the default; this marker file restores the old
// cudaLaunchHostFunc path (the spec's `--cpu-spill-handoff hostfunc|doorbell`, as a file so it needs no flag plumbing).
bool hostfunc_opt_out() {
    static const bool on = [] {
        if (std::FILE* f = std::fopen("C:\\LLM\\cpu-spill-hostfunc.on", "rb")) {
            std::fclose(f);
            return true;
        }
        return false;
    }();
    return on;
}
// [D1] `C:\LLM\cpu-shadow-measure.on` turns on the Q2_0 shadow sample-and-measure pass: it round-trips a strided sample of
// the real spilled NVFP4 expert blocks through the Q2_0 writer and logs the SNR, allocating no shadow and changing nothing.
// (A marker file rather than a flag: `model.flags()` is a LoadRequest, and the option plumbing does not reach it.)
bool shadow_measure_mode() {
    static const bool on = [] {
        if (std::FILE* f = std::fopen("C:\\LLM\\cpu-shadow-measure.on", "rb")) {
            std::fclose(f);
            return true;
        }
        return false;
    }();
    return on;
}
// [D1] The Q2_0 writer.  This fork's Q2_0 is 18 bytes per 64 values: an fp16 scale then 16 bytes of 2-bit codes, four per
// byte, low bits first, and the symbol set is ASYMMETRIC ({-d, 0, +d, +2d}), so the scale must cover -d..+2d and not the
// symmetric amax/2.  Verified against the tree's own dequantize_q2_0: 0 packing mismatches, 0 saturations over 200k
// synthetic blocks (C:\LLM\bwtest\q2_roundtrip.cpp).
static uint16_t f32_to_f16(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) return (uint16_t) (sign | 0x7C00u | (m ? 0x200u : 0));
    if (e >= 0x1F) return (uint16_t) (sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const int shift = 14 - e;
        uint32_t hm = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1u))) ++hm;
        return (uint16_t) (sign | hm);
    }
    uint32_t hm = m >> 13;
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
        if (++hm == 0x400u) { hm = 0; if (++e >= 0x1F) return (uint16_t) (sign | 0x7C00u); }
    }
    return (uint16_t) (sign | ((uint32_t) e << 10) | hm);
}

static void quantize_q2_0_block_scaled(const float* src, uint8_t* dst, float mult) {
    float max_pos = 0.f, max_neg = 0.f;
    for (int j = 0; j < 64; ++j) {
        if (src[j] > max_pos) max_pos = src[j];
        if (src[j] < max_neg) max_neg = src[j];
    }
    float d = std::max(max_neg * -1.f, max_pos * 0.5f) * mult;
    if (d <= 0.f) d = 1.f;
    const float id = 1.f / d;
    const uint16_t half = f32_to_f16(d);
    dst[0] = (uint8_t) (half & 0xFF);
    dst[1] = (uint8_t) (half >> 8);
    uint8_t qs[16] = {};
    for (int j = 0; j < 64; ++j) {
        int c = (int) std::lrintf(src[j] * id) + 1;
        c = std::clamp(c, 0, 3);
        qs[j / 4] |= (uint8_t) (c << (2 * (j % 4)));
    }
    std::memcpy(dst + 2, qs, 16);
}

static void quantize_q2_0_block(const float* src, uint8_t* dst) { quantize_q2_0_block_scaled(src, dst, 1.f); }

// [D1] Round-trip SNR over real spilled blobs: NVFP4 -> the tree's dequant -> the writer -> the tree's Q2_0 dequant.
static double q2_roundtrip_snr(const uint8_t* nv_blob, int n_embd, int n_ff, int blocks_per_part) {
    double sig = 0, err = 0;
    int done = 0;
    const uint8_t* parts[3] = {nv_blob, nv_blob + (size_t) n_ff * (n_embd / 64) * 36,
                               nv_blob + 2 * (size_t) n_ff * (n_embd / 64) * 36};
    const int widths[3] = {n_embd, n_embd, n_ff};
    for (int part = 0; part < 3 && done < blocks_per_part; ++part) {
        const int nb = widths[part] / 64;
        for (int b = 0; b < nb && done < blocks_per_part; ++b, ++done) {
            float v[64], back[64];
            uint8_t q[18];
            basalt::dequantize_nvfp4(parts[part] + (size_t) b * 36, v);
            quantize_q2_0_block(v, q);
            basalt::dequantize_q2_0(q, back);
            for (int j = 0; j < 64; ++j) {
                sig += (double) v[j] * v[j];
                err += (double) (v[j] - back[j]) * (double) (v[j] - back[j]);
            }
        }
    }
    if (sig <= 0 || err <= 0) return 99.0;
    return 10.0 * std::log10(sig / err);
}


std::atomic<uint64_t>& diag_calls() {
    static std::atomic<uint64_t> n{0};
    return n;
}
bool diag_on() {
    static const bool on = std::getenv("BASALT_DIAG_VISIT") != nullptr;
    return on;
}
}  // namespace

struct CpuSpill::Impl {
    const InklingModel* model = nullptr;
    std::unique_ptr<cpu::ExpertPool> pool;
    std::vector<cpu::NativeFmt> formats;
    std::vector<int> group;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr, done = nullptr;
    float* input = nullptr;
    int32_t* ids = nullptr;
    float* output = nullptr;
    float* output_device = nullptr;
    unsigned long long* table = nullptr;
    unsigned long long* table_device = nullptr;
    bool initialized = false, active = false;
    bool shadow_measured = false;   // [D1] the one-shot shadow sample-and-measure pass
    std::atomic<bool> failed{false};
    char error[256] = {};
    alignas(64) std::array<std::array<uint8_t, cpu::kNativeActBytes>, kDecodeRows> acts;
    std::array<cpu::ExpertJobMulti, kPairs> jobs;
    struct Call { Impl* owner; int layer, rows; };
    std::vector<Call> calls;

    // ---- [W2] the doorbell ring ---------------------------------------------------------------------------------
    // One slot per outstanding visit: the normed input and ids the card publishes into mapped pinned memory, and the
    // CPU's rows it writes back there.  Each counter is the host's alias; the *_dev members are the same bytes as the
    // device sees them.
    bool doorbell = false;
    float* ring_x = nullptr;
    int32_t* ring_ids = nullptr;
    float* ring_out = nullptr;
    uint32_t* ring_done = nullptr;
    uint32_t* ring_seq = nullptr;       ///< per slot: the card's publish counter (atomicAdd), mapped
    uint32_t* ring_info = nullptr;      ///< per slot: {layer, rows}, written by the publish kernel
    uint32_t* d_seq = nullptr;          ///< unused by the visit design; kept for the store variant
    float* ring_x_dev = nullptr;
    int32_t* ring_ids_dev = nullptr;
    float* ring_out_dev = nullptr;
    uint32_t* ring_done_dev = nullptr;
    uint32_t* ring_seq_dev = nullptr;
    uint32_t* ring_info_dev = nullptr;
    uint32_t* d_seq_dev = nullptr;
    struct Meta { int layer, rows; };
    std::array<Meta, kRing> meta{};
    std::thread poller;
    std::atomic<bool> stop{false};
    std::atomic<uint32_t> consumed{0};
    uint32_t published = 0, finished = 0;   ///< host-side counts (one engine thread drives start/finish)

    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        stop.store(true, std::memory_order_release);
        if (poller.joinable()) poller.join();
        pool.reset();
        if (table_device) cudaFree(table_device);
        for (void* p : {static_cast<void*>(input), static_cast<void*>(ids), static_cast<void*>(output), static_cast<void*>(table),
                        static_cast<void*>(ring_x), static_cast<void*>(ring_ids), static_cast<void*>(ring_out),
                        static_cast<void*>(ring_done), static_cast<void*>(ring_seq), static_cast<void*>(ring_info), static_cast<void*>(d_seq)})
            if (p) cudaFreeHost(p);
        if (ready) cudaEventDestroy(ready);
        if (done) cudaEventDestroy(done);
        if (stream) cudaStreamDestroy(stream);
    }

    void open(const InklingModel& m, int workers, cpu::PoolAffinity affinity) {
        model = &m;
        const auto& g = m.geo();
        pool = std::make_unique<cpu::ExpertPool>(workers, affinity);
        group.resize(g.n_expert);
        formats.resize(g.n_layers);
        for (int l = 0; l < g.n_layers; ++l) {
            const auto& w = m.layer(l);
            std::string reason;
            if (!cpu::native_fmt(w.gu_type, w.d_type, g.n_embd, g.n_ff_exp, formats[l], reason))
                throw std::runtime_error(reason);
            if (formats[l].bytes > w.blob_bytes) throw std::runtime_error("CPU expert exceeds its backing blob");
            formats[l].gate_softcap = g.gate_softcap;
            formats[l].up_softcap = g.up_softcap;
            for (int rows = 1; rows <= kDecodeRows; ++rows) calls.push_back({this, l, rows});
        }
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checked(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        checked(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        checked(cudaMallocHost(reinterpret_cast<void**>(&input), kDecodeRows * kWidth * sizeof(float)));
        checked(cudaMallocHost(reinterpret_cast<void**>(&ids), kPairs * sizeof(int32_t)));
        checked(cudaHostAlloc(reinterpret_cast<void**>(&output), kPairs * kWidth * sizeof(float), cudaHostAllocMapped));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&output_device), output, 0));
        const size_t bytes = (size_t) g.n_layers * g.n_expert * sizeof(unsigned long long);
        checked(cudaMallocHost(reinterpret_cast<void**>(&table), bytes));
        checked(cudaMalloc(reinterpret_cast<void**>(&table_device), bytes));
        std::memset(table, 0, bytes);

        // [W2] The doorbell hand-off is the default (measured 45.5 -> ~64 tok/s).  The marker file restores the old
        // cudaLaunchHostFunc path for an A/B.
        doorbell = !hostfunc_opt_out();
    }

    /// [W2] Allocated at the first decode visit, never during plan: the load's own pinned read staging is pinned against
    /// the same host-allocation quota (~14.6 GiB here, with the spill using all but ~150 MiB), so a ring pinned at load
    /// time starves the loader.  Every region the card and the CPU both touch is mapped pinned host memory.
    void ensure_ring() {
        if (ring_x) return;
        checked(cudaHostAlloc(reinterpret_cast<void**>(&ring_x), (size_t) kRing * kDecodeRows * kWidth * sizeof(float),
                              cudaHostAllocMapped));
        checked(cudaHostAlloc(reinterpret_cast<void**>(&ring_ids), (size_t) kRing * kPairs * sizeof(int32_t), cudaHostAllocMapped));
        checked(cudaHostAlloc(reinterpret_cast<void**>(&ring_out), (size_t) kRing * kPairs * kWidth * sizeof(float),
                              cudaHostAllocMapped));
        checked(cudaHostAlloc(reinterpret_cast<void**>(&ring_done), (size_t) kRing * sizeof(uint32_t), cudaHostAllocMapped));
        checked(cudaHostAlloc(reinterpret_cast<void**>(&ring_seq), (size_t) kRing * sizeof(uint32_t), cudaHostAllocMapped));
        checked(cudaHostAlloc(reinterpret_cast<void**>(&ring_info), (size_t) kRing * 2 * sizeof(uint32_t), cudaHostAllocMapped));
        checked(cudaHostAlloc(reinterpret_cast<void**>(&d_seq), sizeof(uint32_t), cudaHostAllocMapped));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ring_x_dev), ring_x, 0));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ring_ids_dev), ring_ids, 0));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ring_out_dev), ring_out, 0));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ring_done_dev), ring_done, 0));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ring_seq_dev), ring_seq, 0));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ring_info_dev), ring_info, 0));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_seq_dev), d_seq, 0));
        std::memset(ring_done, 0, (size_t) kRing * sizeof(uint32_t));
        std::memset(ring_seq, 0, (size_t) kRing * sizeof(uint32_t));
        *d_seq = 0;
        log::info("basalt inkling2", "CPU spill: doorbell ring %.1f MiB pinned at the first decode visit",
                  (double) ((size_t) kRing * ((size_t) kDecodeRows * kWidth + kPairs + (size_t) kPairs * kWidth) * sizeof(float)) / (1024.0 * 1024.0));
        poller = std::thread([this] { poll_loop(); });
    }

    /// One visit's CPU work, over explicit buffers so the doorbell path can use its ring slot.
    void run(int layer, int rows, const float* in, const int32_t* ids_in, float* out) {
        std::fill(group.begin(), group.end(), -1);
        std::array<bool, kDecodeRows> quantize{};
        int n = 0;
        for (int p = 0; p < rows * kUsed; ++p) {
            const int e = ids_in[p], t = p / kUsed;
            if (e < 0 || e >= model->geo().n_expert) throw std::runtime_error("CPU expert id out of range");
            if (table[layer * model->geo().n_expert + e]) continue;
            int& at = group[e];
            if (at < 0) {
                at = n++;
                jobs[at] = cpu::ExpertJobMulti{};
                jobs[at].blob = model->host_blob(layer, e);
                if (!jobs[at].blob) throw std::runtime_error("CPU expert has no RAM backing");
            }
            auto& job = jobs[at];
            if (job.nt >= kDecodeRows) throw std::runtime_error("CPU expert has repeated token routing");
            job.nact[job.nt] = acts[t].data();
            job.out[job.nt++] = out + (size_t) p * kWidth;
            quantize[t] = true;
        }
        for (int t = 0; t < rows; ++t)
            if (quantize[t]) cpu::native_quant_act(formats[layer], in + (size_t) t * kWidth, acts[t].data());
        if (n) pool->run_split_multi_native(formats[layer], jobs.data(), n);
    }

    void note_failure(const char* what) {
        std::snprintf(error, sizeof(error), "%s", what);
        failed.store(true, std::memory_order_release);
    }

    /// [W2] The engine thread's replacement for the driver's host-callback dispatch: it reads the card's publish
    /// counter straight out of mapped memory and runs each visit's pool work in order, then writes the rows back and
    /// raises that slot's done word for the card's `wait_flag_ge` node to see.
    void poll_loop() {
        std::array<uint32_t, kRing> seen{};
        while (!stop.load(std::memory_order_acquire)) {
            bool did = false;
            for (int v = 0; v < kRing; ++v) {
                const uint32_t pub = *reinterpret_cast<volatile uint32_t*>(ring_seq + v);
                if (pub == seen[(size_t) v]) continue;
                // what the publish kernel recorded for this visit, so a replayed graph needs no host bookkeeping
                const uint32_t layer = *reinterpret_cast<volatile uint32_t*>(ring_info + 2 * v);
                const uint32_t rows = *reinterpret_cast<volatile uint32_t*>(ring_info + 2 * v + 1);
                try {
                    run((int) layer, (int) rows, ring_x + (size_t) v * kDecodeRows * kWidth,
                        ring_ids + (size_t) v * kPairs, ring_out + (size_t) v * kPairs * kWidth);
                } catch (const std::exception& e) {
                    note_failure(e.what());
                } catch (...) {
                    note_failure("CPU expert execution failed");
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                seen[(size_t) v] = pub;
                *reinterpret_cast<volatile uint32_t*>(ring_done + v) = pub;   // the card's doorbell_wait compares this
                std::atomic_thread_fence(std::memory_order_seq_cst);
                did = true;
            }
            if (!did) platform::cpu_relax();
        }
    }

    static void CUDART_CB execute(void* arg) noexcept {
        if (diag_on()) diag_calls().fetch_add(1, std::memory_order_relaxed);   // [local diag]
        auto& c = *static_cast<Call*>(arg);
        if (c.owner->failed.load(std::memory_order_acquire)) return;
        try {
            c.owner->run(c.layer, c.rows, c.owner->input, c.owner->ids, c.owner->output);
        } catch (const std::exception& e) {
            c.owner->note_failure(e.what());
        } catch (...) {
            c.owner->note_failure("CPU expert execution failed");
        }
    }
};

CpuSpill::CpuSpill() : impl_(std::make_unique<Impl>()) {}
CpuSpill::~CpuSpill() = default;

std::unique_ptr<CpuSpill> CpuSpill::make(const InklingModel& model, std::string& err) {
    // Unknown or wider links keep the card's mapped-RAM path.
    const double link = core::device_rates(model.device()).link_bytes;
    if (!model.spill().any() || model.geo().n_used != kUsed) return nullptr;  // [local] link gate removed: CPU spill wins on this 15.75 GB/s link too
    const auto& flags = model.flags();
    const auto affinity = flags.pool_affinity_given ? flags.pool_affinity : cpu::PoolAffinity::All;
    int workers = flags.pool_workers;
    if (workers <= 0) {
        const auto topo = cpu::detect_cpu_topology(true);
        // [local] every physical core (the hybrid heuristic picks 7 on this 4P+8E chip); the CPU pool feeds the
        // spilled experts and scales with workers.  An explicit --pool-workers still wins.
        workers = cpu::auto_pool_workers((int) topo.worker_cores.size(), topo.p_cores, topo.e_cores, topo.is_hybrid,
                                        cpu::cgroup_cpu_quota(), affinity);
    }
    workers = std::max(1, workers / model.sessions());
    if (workers < 2) return nullptr;
    try {
        auto out = std::unique_ptr<CpuSpill>(new CpuSpill());
        out->impl_->open(model, workers, affinity);
        log::info("basalt inkling2", "CPU spill: %d workers, host link %.2f GB/s, hand-off %s", workers, link / 1e9,
                  out->impl_->doorbell ? "doorbell" : "hostfunc");
        return out;
    } catch (const std::exception& e) {
        err = e.what();
        return nullptr;
    }
}

bool CpuSpill::refresh(void* stream) {
    auto& p = *impl_;
    bool changed = !p.initialized;
    p.active = false;
    const auto& g = p.model->geo();
    for (int l = 0; l < g.n_layers; ++l)
        for (int e = 0; e < g.n_expert; ++e) {
            const auto at = reinterpret_cast<unsigned long long>(p.model->resident_blob(l, e));
            const int i = l * g.n_expert + e;
            if (p.table[i] != at) { p.table[i] = at; changed = true; }
            p.active |= at == 0;
        }
    if (changed) checked(cudaMemcpyAsync(p.table_device, p.table, (size_t) g.n_layers * g.n_expert * 8, cudaMemcpyHostToDevice, (cudaStream_t) stream));
    p.initialized = true;
    // [D1] --cpu-shadow measure: on the first forward after load (the spilled blobs only live in pinned RAM from then on),
    // round-trip a sample of real NVFP4 expert blocks through the Q2_0 writer and log the SNR.  No shadow is allocated and
    // nothing changes: this is the quality evidence a --cpu-shadow q2 decision is made on.
    if (!p.shadow_measured && shadow_measure_mode()) {
        p.shadow_measured = true;
        double sig_all = 0, err_all = 0, err_best_all = 0;
        int layers_done = 0;
        const int widths[3] = {g.n_embd, g.n_embd, g.n_ff_exp};
        for (int l = 0; l < g.n_layers; ++l) {
            const auto& w = p.model->layer(l);
            if (w.gu_type != GGML_TYPE_NVFP4 || w.n_spill == 0) continue;
            double sig = 0, err = 0, err_best = 0;
            int experts = 0;
            for (int e = 0; e < g.n_expert && experts < 4; ++e) {
                const uint8_t* blob = p.model->host_blob(l, e);
                if (!blob) continue;
                ++experts;
                const uint8_t* parts[3] = {blob, blob + (size_t) g.n_ff_exp * (g.n_embd / 64) * 36,
                                           blob + 2 * (size_t) g.n_ff_exp * (g.n_embd / 64) * 36};
                for (int part = 0; part < 3; ++part)
                    for (int b = 0; b < widths[part] / 64; b += 97) {   // a stride through the rows, not the first blocks
                        float v[64], back[64];
                        uint8_t q[18];
                        basalt::dequantize_nvfp4(parts[part] + (size_t) b * 36, v);
                        double best = 1e30;
                        for (float mult : {1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f}) {   // [D1] clipping ladder: d smaller than the
                            quantize_q2_0_block_scaled(v, q, mult);                 // cover bound buys resolution for the bulk
                            basalt::dequantize_q2_0(q, back);
                            double e = 0;
                            for (int j = 0; j < 64; ++j) e += (double) (v[j] - back[j]) * (double) (v[j] - back[j]);
                            if (e < best) best = e;
                        }
                        quantize_q2_0_block(v, q);   // the cover bound, for the like-for-like number
                        basalt::dequantize_q2_0(q, back);
                        double ecov = 0;
                        for (int j = 0; j < 64; ++j) {
                            sig += (double) v[j] * v[j];
                            ecov += (double) (v[j] - back[j]) * (double) (v[j] - back[j]);
                        }
                        err_best += best;
                        for (int j = 0; j < 64; ++j) err += (double) (v[j] - back[j]) * (double) (v[j] - back[j]);
                    }
            }
            if (sig <= 0) continue;
            sig_all += sig;
            err_all += err;
            err_best_all += err_best;
            ++layers_done;
            log::info("basalt inkling2", "cpu-shadow measure: layer %d  Q2_0 round-trip SNR %.2f dB (%d experts sampled)", l,
                      10.0 * std::log10(sig / (err > 0 ? err : 1e-30)), experts);
        }
        log::info("basalt inkling2",
                  "cpu-shadow measure: %d NVFP4 layers, aggregate Q2_0 round-trip SNR %.2f dB (the shadow is OFF: nothing "
                  "changed, --cpu-shadow q2 would use it)",
                  layers_done, err_all > 0 ? 10.0 * std::log10(sig_all / err_all) : 99.0);
        log::info("basalt inkling2", "cpu-shadow measure: with a clipping-ladder scale the aggregate SNR is %.2f dB",
                  err_best_all > 0 ? 10.0 * std::log10(sig_all / err_best_all) : 99.0);
    }
    return p.active;
}

const unsigned long long* CpuSpill::blobs(int layer) const { return impl_->table_device + layer * impl_->model->geo().n_expert; }

void CpuSpill::start(int layer, int rows, const float* input, const int32_t* ids, void* stream) {
    auto& p = *impl_;
    if (p.doorbell) {
        p.ensure_ring();
        // One slot per visit, and the slot index is the visit's position in the window: it is the same on every replay, so a
        // captured graph stays correct.  The publish carries the layer and row count, so the host needs no per-visit call.
        const uint32_t i = p.published % (uint32_t) kRing;
        kernels::doorbell_publish_visit(input, ids, (int64_t) rows * kWidth, (int64_t) rows * kUsed,
                                        p.ring_x_dev + (size_t) i * kDecodeRows * kWidth,
                                        p.ring_ids_dev + (size_t) i * kPairs, p.ring_info_dev + 2 * (size_t) i,
                                        (uint32_t) layer, (uint32_t) rows, p.ring_seq_dev + i, stream);
        ++p.published;
        return;
    }
    checked(cudaEventRecord(p.ready, (cudaStream_t) stream));
    checked(cudaStreamWaitEvent(p.stream, p.ready, 0));
    checked(cudaMemcpyAsync(p.input, input, (size_t) rows * kWidth * 4, cudaMemcpyDeviceToHost, p.stream));
    checked(cudaMemcpyAsync(p.ids, ids, (size_t) rows * kUsed * 4, cudaMemcpyDeviceToHost, p.stream));
    checked(cudaLaunchHostFunc(p.stream, Impl::execute, &p.calls[layer * kDecodeRows + rows - 1]));
    checked(cudaEventRecord(p.done, p.stream));
}

void CpuSpill::finish(int layer, int rows, const int32_t* ids, float* output, uint32_t* tally, void* stream) {
    auto& p = *impl_;
    if (p.doorbell) {
        const uint32_t i = p.finished % (uint32_t) kRing;
        // doorbell_wait compares two mapped words at run time, so a replayed graph is not left waiting on a value that was
        // fixed at capture time (that was what stalled the first attempt).
        kernels::doorbell_wait(p.ring_done_dev + i, p.ring_seq_dev + i, stream);
        moe_cpu_merge(ids, rows, kUsed, blobs(layer), p.ring_out_dev + (size_t) i * kPairs * kWidth, output, tally, stream);
        ++p.finished;
        return;
    }
    checked(cudaStreamWaitEvent((cudaStream_t) stream, p.done, 0));
    moe_cpu_merge(ids, rows, kUsed, blobs(layer), p.output_device, output, tally, stream);
}

bool CpuSpill::ok(std::string& err) const {
    if (!impl_->failed.load(std::memory_order_acquire)) return true;
    err = impl_->error;
    return false;
}

void CpuSpill::diag_pool(double& ms, int64_t& bytes) const {   // [local diag]
    if (!impl_->pool) {
        ms = 0;
        bytes = 0;
        return;
    }
    ms = impl_->pool->ms_multi_gu + impl_->pool->ms_multi_q + impl_->pool->ms_multi_down;
    bytes = impl_->pool->multi_bytes;
}

uint64_t diag_cpu_spill_calls() { return diag_calls().load(std::memory_order_relaxed); }   // [local diag]

}  // namespace basalt::models::inkling2
