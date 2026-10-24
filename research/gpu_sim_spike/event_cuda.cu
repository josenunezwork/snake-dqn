#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// One thread owns one world's ordered stream. Parallel worlds never share state.
__global__ void fifo(int64_t* tags, int64_t* queues, const int64_t* counts,
                     const int64_t* cells, const bool* masks, bool* accepted,
                     int64_t worlds, int64_t board, int64_t n, int64_t qwidth,
                     int64_t old_food, int64_t cap, bool corpse) {
  int64_t e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= worlds) return;
  auto tag = tags + e * board;
  auto queue = queues + e * qwidth;
  int64_t head = 0, count = counts[e];
  for (int64_t col = 0; col < n; ++col) {
    int64_t index = e * n + col, cid = cells[index];
    bool take = masks[index] && tag[cid] == 0;
    accepted[index] = take;
    if (!take) continue;
    if (corpse) {
      bool full = count >= cap;
      if (full) tag[queue[head] - 1] = 0;
      int64_t slot = full ? head : (head + count) % cap;
      queue[slot] = cid + 1;
      if (full) head = (head + 1) % cap;
      else ++count;
      tag[cid] = old_food + col + 1;
    } else tag[cid] = -1;
  }
}

void append_fifo_cuda(torch::Tensor tag, torch::Tensor queue, torch::Tensor count,
                      torch::Tensor cells, torch::Tensor mask, torch::Tensor accepted,
                      int64_t old_food, int64_t cap, bool corpse, bool diagnostics) {
  c10::cuda::CUDAGuard guard(tag.device());
  for (const auto& t : {tag, queue, count, cells, mask, accepted})
    TORCH_CHECK(t.is_cuda() && t.device() == tag.device() && t.is_contiguous(),
                "FIFO tensors must be contiguous on one CUDA device");
  TORCH_CHECK(tag.scalar_type() == torch::kLong && queue.scalar_type() == torch::kLong &&
              count.scalar_type() == torch::kLong && cells.scalar_type() == torch::kLong &&
              mask.scalar_type() == torch::kBool && accepted.scalar_type() == torch::kBool,
              "FIFO dtype mismatch");
  TORCH_CHECK(tag.dim() == 2 && queue.dim() == 2 && cells.dim() == 2 &&
              mask.sizes() == cells.sizes() && accepted.sizes() == cells.sizes() &&
              tag.size(0) == cells.size(0) && queue.size(0) == cells.size(0) &&
              count.numel() == cells.size(0) && (!corpse || (cap > 0 && queue.size(1) == cap)),
              "FIFO shape/cap mismatch");
  if (tag.size(0) == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream(tag.get_device());
  if (diagnostics) {
    // A direct pybind launch lacks dispatcher ancestry.  Keep this CPU scope
    // strictly diagnostic so profiler-free timing is byte-for-byte unchanged.
    RECORD_FUNCTION("gpu_sim.native.event.fifo", std::vector<c10::IValue>());
    fifo<<<(tag.size(0) + 127) / 128, 128, 0, stream.stream()>>>(
        tag.data_ptr<int64_t>(), queue.data_ptr<int64_t>(), count.data_ptr<int64_t>(),
        cells.data_ptr<int64_t>(), mask.data_ptr<bool>(), accepted.data_ptr<bool>(),
        tag.size(0), tag.size(1), cells.size(1), queue.size(1), old_food, cap, corpse);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    fifo<<<(tag.size(0) + 127) / 128, 128, 0, stream.stream()>>>(
        tag.data_ptr<int64_t>(), queue.data_ptr<int64_t>(), count.data_ptr<int64_t>(),
        cells.data_ptr<int64_t>(), mask.data_ptr<bool>(), accepted.data_ptr<bool>(),
        tag.size(0), tag.size(1), cells.size(1), queue.size(1), old_food, cap, corpse);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
