// SPDX-License-Identifier: Apache-2.0
// Persistent CPU packers and a GIL-free, single-stream submission pipeline.
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <pthread.h>

#ifndef EXPERT_STAGING_CPU_TEST
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <acl/acl.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr int64_t kAlignment = 64;
int64_t aligned(int64_t n) { return (n + kAlignment - 1) / kAlignment * kAlignment; }
int64_t ns(Clock::duration d) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

struct Source {
  uintptr_t pointer;
  int64_t bytes;
  at::ScalarType dtype;
  std::vector<int64_t> shape, stride;
};
struct Destination {
  at::Tensor tensor;  // Undefined for raw physical/NZ storage.
  uintptr_t pointer;
  int64_t bytes;
};
struct Copy { int64_t source, destination, offset; };
struct Work {
  std::vector<Copy> copies;
  bool ready = false;
  std::exception_ptr error;
  int64_t pack_ns = 0;
};
struct Chunk {
  int64_t offset, capacity, experts;
  bool in_flight = false;
  std::vector<at::Tensor> references;
#ifndef EXPERT_STAGING_CPU_TEST
  aclrtEvent event = nullptr;
#else
  // CPU tests really defer reading pinned sources until the fake DMA finishes.
  std::thread dma;
  std::exception_ptr dma_error;
#endif
};

class Pipeline {
 public:
  Pipeline(at::Tensor staging, std::vector<int64_t> capacities,
           std::vector<int64_t> experts, int threads, double timeout, int delay_us)
      : staging_(std::move(staging)), timeout_(timeout), delay_us_(delay_us) {
    TORCH_CHECK(staging_.device().is_cpu() && staging_.scalar_type() == at::kByte &&
                staging_.is_contiguous(), "staging must be contiguous CPU bytes");
    TORCH_CHECK(!capacities.empty() && capacities.size() == experts.size(), "invalid chunks");
    TORCH_CHECK(threads > 0 && threads <= 64 && timeout > 0, "invalid workers/timeout");
    int64_t offset = 0;
    for (size_t i = 0; i < capacities.size(); ++i) {
      TORCH_CHECK(capacities[i] > 0 && capacities[i] % kAlignment == 0 && experts[i] > 0,
                  "invalid chunk capacity");
      auto chunk = std::make_unique<Chunk>();
      chunk->offset = offset; chunk->capacity = capacities[i]; chunk->experts = experts[i];
      chunks_.push_back(std::move(chunk)); offset += capacities[i];
    }
    TORCH_CHECK(offset == staging_.numel(), "chunk sizes must match staging budget");
#ifndef EXPERT_STAGING_CPU_TEST
    TORCH_CHECK(staging_.is_pinned(), "native staging requires pinned memory");
    TORCH_CHECK(delay_us == 0, "test DMA delay is CPU-only");
#endif
    // Thread construction failure must join already-created workers.
    try {
      for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { worker(); });
    } catch (...) { stop_workers(); throw; }
  }

  ~Pipeline() {
    stop_workers();
    // Explicit close retires GPU resources while the manager's context lives.
    // A later Python destructor must not recreate a torn-down NPU context.
    if (!closed_) {
      try { synchronize_impl(); destroy_events(); } catch (...) { std::terminate(); }
    }
  }

  int64_t add_source(uintptr_t ptr, int64_t bytes, at::Tensor prototype,
                     std::vector<int64_t> shape, std::vector<int64_t> stride) {
    auto lock = exclusive();
#ifdef EXPERT_STAGING_CPU_TEST
    synchronize_impl();
#endif
    TORCH_CHECK(!closed_ && !poisoned_ && ptr && bytes > 0, "invalid source or closed pipeline");
    TORCH_CHECK(prototype.device().is_cpu(), "source metadata must be CPU");
    TORCH_CHECK(shape.size() == stride.size(), "invalid source shape/stride");
    int64_t span = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      TORCH_CHECK(shape[i] > 0 && stride[i] >= 0, "invalid source dimensions");
      TORCH_CHECK(shape[i] == 1 || stride[i] <= (bytes / prototype.element_size()) / (shape[i] - 1),
                  "source stride exceeds field");
      span += (shape[i] - 1) * stride[i];
    }
    TORCH_CHECK(span <= bytes / prototype.element_size(), "source view exceeds field");
    sources_.push_back({ptr, bytes, prototype.scalar_type(), std::move(shape), std::move(stride)});
    return sources_.size() - 1;
  }

  int64_t add_destination(at::Tensor tensor, uintptr_t ptr, int64_t bytes) {
    auto lock = exclusive();
#ifdef EXPERT_STAGING_CPU_TEST
    synchronize_impl();
#endif
    if (ptr) tensor = at::Tensor();
    TORCH_CHECK(!closed_ && !poisoned_ && bytes > 0 && (tensor.defined() || ptr), "invalid destination");
#ifndef EXPERT_STAGING_CPU_TEST
    if (tensor.defined()) TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                                    "logical destination must be NPU");
#endif
    destinations_.push_back({std::move(tensor), ptr, bytes});
    return destinations_.size() - 1;
  }

  void run(const std::vector<int64_t>& source_ids, const std::vector<int64_t>& dest_ids,
           const std::vector<int64_t>& ends, int device) {
    auto lock = exclusive();
    TORCH_CHECK(!closed_ && !poisoned_, "native pipeline is closed or poisoned");
    TORCH_CHECK(source_ids.size() == dest_ids.size(), "descriptor size mismatch");
    TORCH_CHECK((ends.empty() && source_ids.empty()) ||
                (!ends.empty() && ends.back() == static_cast<int64_t>(source_ids.size())), "invalid expert ends");
    // Fully validate before any packer or DMA sees a descriptor.
    int64_t previous = 0;
    for (auto end : ends) { TORCH_CHECK(end > previous, "invalid expert boundary"); previous = end; }
    for (size_t i = 0; i < source_ids.size(); ++i) {
      TORCH_CHECK(source_ids[i] >= 0 && source_ids[i] < static_cast<int64_t>(sources_.size()) &&
                  dest_ids[i] >= 0 && dest_ids[i] < static_cast<int64_t>(destinations_.size()), "invalid descriptor ID");
      const auto& s = sources_[source_ids[i]]; const auto& d = destinations_[dest_ids[i]];
      TORCH_CHECK(s.bytes == d.bytes, "source/destination physical byte mismatch");
    }
    stats_.assign(6, 0);
    if (ends.empty()) return;
#ifndef EXPERT_STAGING_CPU_TEST
    TORCH_CHECK(device_ < 0 || device_ == device, "native staging cannot change NPU device");
    device_ = device;
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(device);
    // The caller's persistent TransferExecutor owns this current load stream.
    stream_ = stream.stream(true);  // Drain torch-npu's host queue before raw ACL.
#else
    (void)device;
#endif
    auto start = Clock::now();
    try {
      // Build bounded windows, rather than allocate staging or jobs per layer.
      size_t expert = 0, submitted = 0;
      std::deque<std::pair<size_t, std::shared_ptr<Work>>> pending;
      auto schedule = [&](size_t slot) {
        auto& chunk = *chunks_[slot];
        retire(chunk);
        auto work = std::make_shared<Work>();
        int64_t offset = 0;
        size_t stop = std::min(ends.size(), expert + static_cast<size_t>(chunk.experts));
        for (; expert < stop; ++expert) {
          int64_t begin = expert ? ends[expert - 1] : 0;
          for (int64_t i = begin; i < ends[expert]; ++i) {
            const auto& s = sources_[source_ids[i]];
            TORCH_CHECK(s.bytes <= chunk.capacity - offset, "expert exceeds native staging chunk");
            work->copies.push_back({source_ids[i], dest_ids[i], chunk.offset + offset});
            offset += aligned(s.bytes);
          }
        }
        {
          std::lock_guard<std::mutex> guard(mutex_);
          jobs_.push_back(work);
        }
        condition_.notify_one();
        pending.emplace_back(slot, std::move(work));
      };
      const size_t first_chunk = next_chunk_;
      for (size_t i = 0; i < chunks_.size() && expert < ends.size(); ++i)
        schedule((first_chunk + i) % chunks_.size());
      while (!pending.empty()) {
        auto [slot, work] = pending.front(); pending.pop_front();
        auto wait_start = Clock::now();
        {
          std::unique_lock<std::mutex> guard(mutex_);
          TORCH_CHECK(condition_.wait_for(guard, std::chrono::duration<double>(timeout_),
                                         [&] { return work->ready; }), "native pack timeout");
        }
        stats_[1] += ns(Clock::now() - wait_start);
        if (work->error) std::rethrow_exception(work->error);
        stats_[0] += work->pack_ns;
        auto submit_start = Clock::now();
        submit(*chunks_[slot], work);
        next_chunk_ = (slot + 1) % chunks_.size();
        if (!submitted++) stats_[3] = ns(submit_start - start);
        stats_[2] += ns(Clock::now() - submit_start);
        if (expert < ends.size()) schedule(slot);
      }
      stats_[4] = ns(Clock::now() - start); stats_[5] = submitted;
    } catch (...) {
      // Futures own their descriptors; wait for all packers before a caller can
      // close canonical mappings. Never silently retry an in-flight DMA.
      poisoned_ = true;
      wait_packers();
      throw;
    }
  }

  void synchronize() { auto lock = exclusive(); wait_packers(); synchronize_impl(); }
  void close() {
    auto lock = exclusive();
    if (closed_) return;
    wait_packers(); synchronize_impl(); stop_workers(); destroy_events();
    destinations_.clear(); sources_.clear(); closed_ = true;
  }
  std::vector<int64_t> stats() { auto lock = exclusive(); return stats_; }

 private:
  std::unique_lock<std::mutex> exclusive() {
    std::unique_lock<std::mutex> lock(api_mutex_, std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(), "overlapping native staging calls"); return lock;
  }
  void worker() {
    (void)pthread_setname_np(pthread_self(), "expert-pack");
    for (;;) {
      std::shared_ptr<Work> work;
      {
        std::unique_lock<std::mutex> guard(mutex_);
        condition_.wait(guard, [&] { return stop_ || !jobs_.empty(); });
        if (stop_ && jobs_.empty()) return;
        work = jobs_.front(); jobs_.pop_front(); ++active_;
      }
      auto start = Clock::now();
      try {
        auto* base = staging_.data_ptr<uint8_t>();
        for (const auto& copy : work->copies) {
          const auto& s = sources_[copy.source];
          std::memcpy(base + copy.offset, reinterpret_cast<const void*>(s.pointer), s.bytes);
        }
      } catch (...) { work->error = std::current_exception(); }
      {
        std::lock_guard<std::mutex> guard(mutex_);
        work->pack_ns = ns(Clock::now() - start); work->ready = true; --active_;
      }
      condition_.notify_all();
    }
  }
  void wait_packers() {
    std::unique_lock<std::mutex> guard(mutex_);
    condition_.wait(guard, [&] { return jobs_.empty() && active_ == 0; });
  }
  void stop_workers() {
    { std::lock_guard<std::mutex> guard(mutex_); stop_ = true; }
    condition_.notify_all();
    for (auto& t : workers_) if (t.joinable()) t.join();
  }
  void retire(Chunk& chunk) {
    if (chunk.in_flight) {
#ifndef EXPERT_STAGING_CPU_TEST
      aclrtEventRecordedStatus status;
      check(aclrtQueryEventStatus(chunk.event, &status), "query staging event");
      if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE) check(aclrtSynchronizeEvent(chunk.event), "wait staging event");
#else
      if (chunk.dma.joinable()) chunk.dma.join();
      if (chunk.dma_error) {
        auto error = chunk.dma_error; chunk.dma_error = nullptr;
        chunk.in_flight = false; chunk.references.clear(); poisoned_ = true;
        std::rethrow_exception(error);
      }
#endif
      chunk.in_flight = false;
    }
    chunk.references.clear();
  }
  void copy_fields(Chunk& chunk, const Work& work) {
    for (const auto& copy : work.copies) {
      const auto& s = sources_[copy.source]; const auto& d = destinations_[copy.destination];
      if (d.tensor.defined()) {
        // Preserve dtype, physical strides, reshape and destination conversion.
        auto cpu = staging_.narrow(0, copy.offset, s.bytes).view(s.dtype)
                       .as_strided(s.shape, s.stride).reshape(d.tensor.sizes());
        chunk.references.push_back(cpu);
        d.tensor.copy_(cpu, true);
#ifndef EXPERT_STAGING_CPU_TEST
        launch_queue_dirty_ = true;
#endif
      } else {
#ifndef EXPERT_STAGING_CPU_TEST
        // copy_ above can be queued on torch-npu's CPU launch queue. Empty that
        // queue before issuing the next raw ACL copy on the same hardware stream.
        if (launch_queue_dirty_) {
          stream_ = c10_npu::getCurrentNPUStream().stream(true);
          launch_queue_dirty_ = false;
        }
        check(aclrtMemcpyAsync(reinterpret_cast<void*>(d.pointer), d.bytes,
                               staging_.data_ptr<uint8_t>() + copy.offset, s.bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream_), "raw expert H2D");
#else
        std::memcpy(reinterpret_cast<void*>(d.pointer), staging_.data_ptr<uint8_t>() + copy.offset, s.bytes);
#endif
      }
    }
  }
  void submit(Chunk& chunk, const std::shared_ptr<Work>& work) {
#ifndef EXPERT_STAGING_CPU_TEST
    if (!chunk.event) check(aclrtCreateEventWithFlag(&chunk.event, ACL_EVENT_SYNC), "create staging event");
    try {
      copy_fields(chunk, *work);
    } catch (...) {
      // Covers partially queued quant operations and direct raw copies.
      stream_ = c10_npu::getCurrentNPUStream().stream(true);
      check(aclrtSynchronizeStream(stream_), "drain failed staging submission");
      throw;
    }
    stream_ = c10_npu::getCurrentNPUStream().stream(true);
    launch_queue_dirty_ = false;
    // Mark in-flight before record: a record failure must still drain on close.
    chunk.in_flight = true;
    if (aclrtRecordEvent(chunk.event, stream_) != ACL_SUCCESS) {
      check(aclrtSynchronizeStream(stream_), "drain failed event record");
      chunk.in_flight = false;
      TORCH_CHECK(false, "record staging event failed");
    }
#else
    chunk.dma_error = nullptr;
    // Model one ordered DMA stream, including overlapping destination slots.
    // CPU copies must not run concurrently merely because chunks differ.
    auto previous = last_dma_;
    auto done = std::make_shared<std::promise<void>>();
    last_dma_ = done->get_future().share();
    chunk.dma = std::thread([this, &chunk, work, previous, done] {
      try {
        if (previous.valid()) previous.wait();
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us_));
        copy_fields(chunk, *work);
      } catch (...) { chunk.dma_error = std::current_exception(); }
      done->set_value();
    });
    chunk.in_flight = true;
#endif
  }
  void synchronize_impl() {
#ifndef EXPERT_STAGING_CPU_TEST
    c10_npu::OptionalNPUGuard guard;
    if (device_ >= 0) guard.set_index(device_);
#endif
    std::exception_ptr error;
    for (auto& c : chunks_) {
      try { retire(*c); } catch (...) { if (!error) error = std::current_exception(); }
    }
    if (error) std::rethrow_exception(error);
  }
  void destroy_events() {
#ifndef EXPERT_STAGING_CPU_TEST
    c10_npu::OptionalNPUGuard guard;
    if (device_ >= 0) guard.set_index(device_);
    for (auto& c : chunks_) {
      if (c->event) { check(aclrtDestroyEvent(c->event), "destroy staging event"); c->event = nullptr; }
    }
#endif
  }
#ifndef EXPERT_STAGING_CPU_TEST
  static void check(aclError code, const char* what) { TORCH_CHECK(code == ACL_SUCCESS, what, ": ", code); }
  aclrtStream stream_ = nullptr;
  int device_ = -1;
  bool launch_queue_dirty_ = false;
#else
  std::shared_future<void> last_dma_;
#endif
  at::Tensor staging_;
  std::vector<Source> sources_;
  std::vector<Destination> destinations_;
  std::vector<std::unique_ptr<Chunk>> chunks_;
  std::vector<std::thread> workers_;
  std::deque<std::shared_ptr<Work>> jobs_;
  std::mutex api_mutex_, mutex_;
  std::condition_variable condition_;
  size_t active_ = 0;
  size_t next_chunk_ = 0;
  bool stop_ = false, closed_ = false, poisoned_ = false;
  double timeout_;
  int delay_us_;
  std::vector<int64_t> stats_;
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  py::class_<Pipeline>(m, "Pipeline", py::module_local())
      .def(py::init<at::Tensor, std::vector<int64_t>, std::vector<int64_t>, int, double, int>())
      .def("add_source", &Pipeline::add_source, py::call_guard<py::gil_scoped_release>())
      .def("add_destination", &Pipeline::add_destination, py::call_guard<py::gil_scoped_release>())
      .def("run", &Pipeline::run, py::call_guard<py::gil_scoped_release>())
      .def("synchronize", &Pipeline::synchronize, py::call_guard<py::gil_scoped_release>())
      .def("close", &Pipeline::close, py::call_guard<py::gil_scoped_release>())
      .def("stats", &Pipeline::stats, py::call_guard<py::gil_scoped_release>());
}
