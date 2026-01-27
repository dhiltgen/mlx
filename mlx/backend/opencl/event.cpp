// Copyright © 2025 MLX Contributors
// OpenCL Event implementation - using CPU-style synchronization for now

#include "mlx/event.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/scheduler.h"

#include <condition_variable>
#include <mutex>

namespace mlx::core {

// Event counter structure for CPU-style synchronization
struct EventCounter {
  uint64_t value{0};
  std::mutex mtx;
  std::condition_variable cv;
};

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) { delete static_cast<EventCounter*>(ptr); };
  event_ = std::shared_ptr<void>(new EventCounter{}, dtor);
}

void Event::wait() {
  auto ec = static_cast<EventCounter*>(event_.get());
  std::unique_lock<std::mutex> lk(ec->mtx);
  if (ec->value >= value()) {
    return;
  }
  ec->cv.wait(lk, [value = value(), ec] { return ec->value >= value; });
}

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable { wait(); });
  } else {
    // For GPU streams, also use the same approach for now
    // TODO: Use OpenCL event dependencies
    scheduler::enqueue(default_stream(Device::cpu), [*this]() mutable { wait(); });
  }
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable {
      auto ec = static_cast<EventCounter*>(event_.get());
      {
        std::lock_guard<std::mutex> lk(ec->mtx);
        ec->value = value();
      }
      ec->cv.notify_all();
    });
  } else {
    // For GPU streams, signal via CPU for now
    // TODO: Use OpenCL event signaling
    scheduler::enqueue(default_stream(Device::cpu), [*this]() mutable {
      auto ec = static_cast<EventCounter*>(event_.get());
      {
        std::lock_guard<std::mutex> lk(ec->mtx);
        ec->value = value();
      }
      ec->cv.notify_all();
    });
  }
}

bool Event::is_signaled() const {
  auto ec = static_cast<EventCounter*>(event_.get());
  {
    std::lock_guard<std::mutex> lk(ec->mtx);
    return (ec->value >= value());
  }
}

} // namespace mlx::core
