#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>
#include <vector>

namespace {
void check_tensor(const torch::Tensor& tensor, const torch::Device& device,
                  torch::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device, name, " must share the CUDA device");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.is_contiguous(), name,
              " dtype/contiguity mismatch");
}

__device__ __forceinline__ int wrap_ring(int value, int cap) {
  value %= cap;
  return value < 0 ? value + cap : value;
}

__global__ void construct_indexed(
    const int64_t* bodies, const int64_t* head_ptr, const int64_t* seg_count,
    const int64_t* length, const bool* alive, const int64_t* direction,
    const int64_t* boost_frames, const int64_t* rows, int* all_occ,
    int* other_occ, int* post_tail, int64_t* release,
    int64_t* lpost_out, bool* boosted_out, int* status, int nr, int worlds,
    int slots, int cap, int h, int w, int min_boost, int boost_cost) {
  const int row = blockIdx.x;
  if (row >= nr) return;
  const int64_t env64 = rows[row * 2], observer64 = rows[row * 2 + 1];
  if (env64 < 0 || env64 >= worlds || observer64 < 0 || observer64 >= slots) {
    if (threadIdx.x == 0) atomicOr(status, 1);
    return;
  }
  const int env = static_cast<int>(env64), observer = static_cast<int>(observer64);
  const int vertices = h * w;
  const int64_t row_base = static_cast<int64_t>(row) * slots;
  const int64_t observer_index = row_base + observer;
  const int64_t sim_base = static_cast<int64_t>(env) * slots;

  // Every thread walks a disjoint (slot, ring-index) range.  OR/max are
  // intentionally idempotent, preserving union occupancy and duplicate-cell
  // maximum release semantics of the frozen tensor implementation.
  for (int linear = threadIdx.x; linear < slots * cap; linear += blockDim.x) {
    const int slot = linear / cap, k = linear % cap;
    const int64_t snake = sim_base + slot;
    if (!alive[snake]) continue;
    if (head_ptr[snake] < 0 || head_ptr[snake] >= cap || seg_count[snake] < 0 ||
        seg_count[snake] > cap) {
      atomicOr(status, 2);
      continue;
    }
    if (k >= seg_count[snake]) continue;
    const int ring = wrap_ring(static_cast<int>(head_ptr[snake]) - k, cap);
    const int64_t body_base = (snake * cap + ring) * 2;
    const int x = static_cast<int>(bodies[body_base]), y = static_cast<int>(bodies[body_base + 1]);
    if (x < 0 || x >= w || y < 0 || y >= h) continue;
    const int cell = y * w + x;
    atomicOr(all_occ + static_cast<int64_t>(row) * vertices + cell, 1);
    if (slot != observer)
      atomicOr(other_occ + static_cast<int64_t>(row) * vertices + cell, 1);
  }

  const int64_t controlled = sim_base + observer;
  const int count = static_cast<int>(seg_count[controlled]);
  const int body_length = static_cast<int>(length[controlled]);
  const int heading = static_cast<int>(direction[controlled]);
  const int bf = static_cast<int>(boost_frames[controlled]);
  const int hp = static_cast<int>(head_ptr[controlled]);
  if (hp < 0 || hp >= cap || count < 0 || count > cap || body_length < 1 ||
      heading < 0 || heading > 3 || bf < 0) {
    if (threadIdx.x == 0) atomicOr(status, 4);
    return;
  }
  const int head_ring = wrap_ring(hp, cap);
  const int64_t head_base = (controlled * cap + head_ring) * 2;
  const int hx = static_cast<int>(bodies[head_base]), hy = static_cast<int>(bodies[head_base + 1]);
  const int dx[4] = {0, 1, 0, -1}, dy[4] = {-1, 0, 1, 0};
  for (int linear = threadIdx.x; linear < 6 * cap; linear += blockDim.x) {
    const int action = linear / cap, i = linear % cap;
    const bool boosted = action >= 3 && body_length >= min_boost;
    int ln = min(count + 1, body_length);
    if (boosted) {
      ln = min(ln + 1, body_length);
    }
    int post_length = body_length;
    if (boosted && bf + 1 >= boost_cost && body_length > 1) {
      --post_length;
      ln = min(ln, post_length);
    }
    if (i == 0) {
      lpost_out[row * 6 + action] = post_length;
      boosted_out[row * 6 + action] = boosted;
    }
    if (i >= ln) continue;
    const int hd = (heading + (action % 3) + 3) % 4;
    const int nh = boosted ? 2 : 1;
    int x, y;
    if (i < nh) {
      const int step = nh - i;
      x = hx + dx[hd] * step;
      y = hy + dy[hd] * step;
    } else {
      const int ring = wrap_ring(hp - (i - nh), cap);
      const int64_t body_base = (controlled * cap + ring) * 2;
      x = static_cast<int>(bodies[body_base]);
      y = static_cast<int>(bodies[body_base + 1]);
    }
    if (x < 0 || x >= w || y < 0 || y >= h) continue;
    const int64_t lane = static_cast<int64_t>(row) * 6 + action;
    const int cell = y * w + x;
    if (i > 0)
      atomicOr(post_tail + lane * vertices + cell, 1);
    const int64_t when = post_length - i + 1;
    atomicMax(reinterpret_cast<unsigned long long*>(release + lane * vertices + cell),
              static_cast<unsigned long long>(when));
  }
}
}  // namespace

std::vector<torch::Tensor> v8_indexed_construct_cuda(
    torch::Tensor bodies, torch::Tensor head_ptr, torch::Tensor seg_count,
    torch::Tensor length, torch::Tensor alive, torch::Tensor direction,
    torch::Tensor boost_frames, torch::Tensor rows, int64_t h, int64_t w,
    int64_t min_boost_length, int64_t boost_length_cost_frames) {
  c10::cuda::CUDAGuard guard(bodies.device());
  const auto device = bodies.device();
  check_tensor(bodies, device, torch::kLong, "bodies");
  check_tensor(head_ptr, device, torch::kLong, "head_ptr");
  check_tensor(seg_count, device, torch::kLong, "seg_count");
  check_tensor(length, device, torch::kLong, "length");
  check_tensor(alive, device, torch::kBool, "alive");
  check_tensor(direction, device, torch::kLong, "direction");
  check_tensor(boost_frames, device, torch::kLong, "boost_frames");
  check_tensor(rows, device, torch::kLong, "rows");
  TORCH_CHECK(bodies.dim() == 4 && bodies.size(3) == 2 && rows.dim() == 2 && rows.size(1) == 2,
              "invalid v8 indexed shapes");
  const auto worlds = bodies.size(0), slots = bodies.size(1), cap = bodies.size(2), nr = rows.size(0);
  TORCH_CHECK(head_ptr.dim() == 2 && head_ptr.size(0) == worlds && head_ptr.size(1) == slots &&
              seg_count.sizes() == head_ptr.sizes() && length.sizes() == head_ptr.sizes() &&
              alive.sizes() == head_ptr.sizes() && direction.sizes() == head_ptr.sizes() &&
              boost_frames.sizes() == head_ptr.sizes(), "v8 indexed state shape mismatch");
  TORCH_CHECK(h > 0 && w > 0 && h * w <= INT_MAX && cap > 0 && nr <= INT_MAX &&
              min_boost_length > 0 && boost_length_cost_frames > 0, "unsupported v8 indexed dimensions");
  const auto vertices = h * w;
  // CUDA only defines integer atomicOr overloads at 32 bits and wider. The
  // public Python boundary converts these idempotent union maps to bool.
  auto byte_options = bodies.options().dtype(torch::kInt32);
  auto long_options = bodies.options().dtype(torch::kLong);
  auto all_occ = torch::zeros({nr, h, w}, byte_options);
  auto other_occ = torch::zeros_like(all_occ);
  auto post_tail = torch::zeros({nr, 6, h, w}, byte_options);
  auto release = torch::zeros({nr, 6, h, w}, long_options);
  auto lpost = torch::zeros({nr, 6}, long_options);
  auto boosted = torch::zeros({nr, 6}, bodies.options().dtype(torch::kBool));
  auto status = torch::zeros({}, bodies.options().dtype(torch::kInt32));
  if (nr == 0) return {all_occ, other_occ, post_tail, release, lpost, boosted, status};
  auto stream = at::cuda::getCurrentCUDAStream(bodies.get_device());
  construct_indexed<<<nr, 256, 0, stream.stream()>>>(
      bodies.data_ptr<int64_t>(), head_ptr.data_ptr<int64_t>(), seg_count.data_ptr<int64_t>(),
      length.data_ptr<int64_t>(), alive.data_ptr<bool>(), direction.data_ptr<int64_t>(),
      boost_frames.data_ptr<int64_t>(), rows.data_ptr<int64_t>(), all_occ.data_ptr<int>(),
      other_occ.data_ptr<int>(), post_tail.data_ptr<int>(), release.data_ptr<int64_t>(),
      lpost.data_ptr<int64_t>(), boosted.data_ptr<bool>(), status.data_ptr<int>(), static_cast<int>(nr),
      static_cast<int>(worlds), static_cast<int>(slots), static_cast<int>(cap), static_cast<int>(h),
      static_cast<int>(w), static_cast<int>(min_boost_length), static_cast<int>(boost_length_cost_frames));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {all_occ, other_occ, post_tail, release, lpost, boosted, status};
}
