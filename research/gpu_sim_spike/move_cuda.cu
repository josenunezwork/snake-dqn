// MOVE: no fast math, indexed body inspection and sequential per-world decisions.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {
constexpr int THREADS = 256;
__device__ int64_t ring_index(int64_t ptr, int64_t k, int64_t cap) {
  int64_t r = (ptr - k) % cap;
  return r < 0 ? r + cap : r;
}
__device__ bool cell_equal(const int64_t* a, const int64_t* b) {
  return a[0] == b[0] && a[1] == b[1];
}

__global__ void respawn_kernel(
    int64_t* bodies, int64_t* ptr, int64_t* count, int64_t* length,
    bool* alive, int64_t* direction, int64_t* boost, int64_t* hunger,
    int64_t* timer, int64_t* prev, const bool* terminal,
    int64_t* counter, const int64_t* key, int32_t* occ,
    int S, int64_t C, int W, int H) {
  const int e = blockIdx.x, lane = threadIdx.x;
  const int64_t row = int64_t(e) * S, board = int64_t(W) * H;
  __shared__ bool needs_placement;
  if (lane == 0) {
    needs_placement = false;
    for (int j = 0; j < S; ++j) {
      if (!alive[row+j] && timer[row+j] > 0) --timer[row+j];
      if (!alive[row+j] && timer[row+j] <= 0 && !terminal[row+j])
        needs_placement = true;
    }
    // Unsigned arithmetic specifies the frozen tensor int64 wrap exactly.
    if (!needs_placement) counter[e] = int64_t(uint64_t(counter[e]) + 32ULL);
  }
  __syncthreads();
  if (!needs_placement) return;
  int32_t* grid = occ + int64_t(e) * board;
  for (int64_t p = lane; p < board; p += blockDim.x) grid[p] = 0;
  __syncthreads();
  for (int j = 0; j < S; ++j) {
    const int64_t r = row+j;
    if (!alive[r]) continue;
    // Inspect actual valid segments, never capacity-sized padding.
    for (int64_t k = lane; k < count[r] && k < C; k += blockDim.x) {
      const int64_t* cell = bodies + (r*C + ring_index(ptr[r], k, C))*2;
      if (cell[0] >= 0 && cell[0] < W && cell[1] >= 0 && cell[1] < H)
        atomicExch(grid + cell[1]*W + cell[0], 1);
    }
  }
  __syncthreads();
  if (lane != 0) return;
  for (int trial = 0; trial < 32; ++trial) {
    counter[e] = int64_t(uint64_t(counter[e]) + 1ULL);
    const uint64_t z = (uint64_t(key[e])*6364136223846793005ULL +
        uint64_t(counter[e])*1442695040888963407ULL) & 0x7fffffffffffffffULL;
    const int bx = z % W, by = (z / W) % H;
    for (int j = 0; j < S; ++j) {
      const int64_t r = row+j;
      if (alive[r] || timer[r] > 0 || terminal[r]) continue;
      const int x = (bx+j)%W, y = (by+j/W)%H;
      if (grid[int64_t(y)*W+x]) continue;
      bool unique = true;
      // Earlier candidates exclude this slot regardless of readiness/success.
      for (int a = 0; a < j; ++a)
        if ((bx+a)%W == x && (by+a/W)%H == y) { unique = false; break; }
      if (!unique) continue;
      bodies[(r*C)*2] = x; bodies[(r*C)*2+1] = y;
      ptr[r] = 0; count[r] = 1; length[r] = 1; prev[r] = 1;
      alive[r] = true; direction[r] = 1;
      boost[r] = 0; hunger[r] = 0; timer[r] = 0;
      grid[int64_t(y)*W+x] = 1;
    }
  }
}

__global__ void inspect_kernel(
    const int64_t* bodies, const int64_t* ptr, const int64_t* count,
    const bool* alive, const int64_t* head0, const int64_t* path,
    const bool* valid, int32_t* flags, int S, int64_t C) {
  const int pair = blockIdx.x, j = pair % S, i = (pair/S)%S, e = pair/(S*S);
  const int lane = threadIdx.x;
  const int64_t ri = int64_t(e)*S+i, rj = int64_t(e)*S+j;
  unsigned hit = 0;
  if (alive[rj]) {
    for (int64_t k = lane; k < count[rj] && k < C; k += blockDim.x) {
      if (i == j && k < 3) continue;
      const int64_t* cell = bodies + (rj*C + ring_index(ptr[rj], k, C))*2;
      for (int t = 0; t < 2; ++t)
        if (valid[ri*2+t] && cell_equal(cell, path+(ri*2+t)*2)) hit = 1;
    }
  }
  __shared__ unsigned shared[THREADS];
  shared[lane] = hit;
  __syncthreads();
  for (int d = THREADS/2; d; d /= 2) {
    if (lane < d) shared[lane] |= shared[lane+d];
    __syncthreads();
  }
  if (lane == 0) {
    bool head = false;
    if (i != j) {
      for (int t = 0; t < 2; ++t) for (int u = 0; u < 2; ++u) {
        head |= valid[ri*2+t] && valid[rj*2+u] &&
            cell_equal(path+(ri*2+t)*2, path+(rj*2+u)*2);
        // The frozen swap test deliberately has no traversal-valid predicate.
        head |= cell_equal(head0+ri*2, path+(rj*2+u)*2) &&
            cell_equal(path+(ri*2+t)*2, head0+rj*2);
      }
    }
    flags[pair*2] = shared[0]; flags[pair*2+1] = head;
  }
}

__global__ void resolve_kernel(
    const int64_t* length, const bool* alive, const bool* arena,
    const int32_t* flags, int64_t* death, int64_t* rank_out,
    bool* dead, int64_t* kills, int64_t* mass, int E, int S, bool v2, float ratio) {
  const int e = blockIdx.x*blockDim.x + threadIdx.x;
  if (e >= E) return;
  const int64_t row = int64_t(e)*S;
  for (int j = 0; j < S; ++j) {
    death[row+j] = 0; rank_out[row+j] = S*S+S;
    dead[row+j] = false; kills[row+j] = 0; mass[row+j] = 0;
  }
  int rank = 0;
  for (int i = 0; i < S; ++i) {
    const int64_t ri = row+i;
    if (alive[ri] && !arena[ri] && !dead[ri]) {
      death[ri] = 1; rank_out[ri] = rank; dead[ri] = true;
    }
    ++rank;
    if (alive[ri] && flags[(int64_t(e)*S*S+i*S+i)*2] && !dead[ri]) {
      death[ri] = 2; rank_out[ri] = rank; dead[ri] = true;
    }
    ++rank;
    for (int j = 0; j < S; ++j) {
      if (i == j) continue;
      const int64_t rj = row+j, pair = int64_t(e)*S*S+i*S+j;
      const bool eligible = alive[ri] && !dead[ri];
      const bool head = eligible && alive[rj] && !dead[rj] && flags[pair*2+1];
      const bool body = eligible && (v2 || (alive[rj] && !dead[rj])) &&
          !flags[pair*2+1] && flags[pair*2];
      const bool iw = head && v2 && float(length[ri]) >= ratio*float(length[rj]);
      const bool jw = head && v2 && float(length[rj]) >= ratio*float(length[ri]);
      const bool mutual = head && !iw && !jw;
      if (body || mutual || jw) {
        death[ri] = body ? 4 : 3; rank_out[ri] = rank; dead[ri] = true;
      }
      if (mutual || iw) { death[rj] = 3; rank_out[rj] = rank; dead[rj] = true; }
      kills[ri] += iw; kills[rj] += jw || body;
      if (iw) mass[ri] += length[rj];
      if (jw || body) mass[rj] += length[ri];
      ++rank;
    }
  }
}

void check(const std::vector<torch::Tensor>& s, size_t n) {
  TORCH_CHECK(s.size() == n, "MOVE state arity mismatch");
  for (const auto& t : s) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "MOVE requires contiguous CUDA state");
    TORCH_CHECK(t.device() == s[0].device(), "MOVE mixed devices");
  }
  TORCH_CHECK(s[0].scalar_type() == torch::kInt64 && s[0].dim() == 4 &&
              s[0].size(3) == 2 && s[0].size(0) > 0 && s[0].size(1) > 0 &&
              s[0].size(2) > 0, "MOVE bodies must be nonempty E,S,C,2 int64");
}
void shape(const torch::Tensor& t, at::ScalarType dtype, at::IntArrayRef dims) {
  TORCH_CHECK(t.scalar_type() == dtype && t.sizes().equals(dims),
              "MOVE state shape/dtype mismatch");
}
}  // namespace

void move_respawn_cuda(std::vector<torch::Tensor> s, torch::Tensor occupancy,
                       int64_t width, int64_t height) {
  check(s, 13);
  const int64_t E = s[0].size(0), S = s[0].size(1), C = s[0].size(2);
  TORCH_CHECK(width > 0 && height > 0 && width <= INT32_MAX && height <= INT32_MAX,
              "MOVE invalid board");
  for (int a = 1; a <= 10; ++a)
    shape(s[a], (a == 4 || a == 10) ? torch::kBool : torch::kInt64, {E,S});
  shape(s[11], torch::kInt64, {E}); shape(s[12], torch::kInt64, {E});
  TORCH_CHECK(occupancy.is_cuda() && occupancy.is_contiguous() &&
              occupancy.device() == s[0].device(), "MOVE invalid occupancy device");
  shape(occupancy, torch::kInt32, {E,width*height});
  const c10::cuda::CUDAGuard guard(s[0].device());
  const auto stream = at::cuda::getCurrentCUDAStream(s[0].get_device());
  respawn_kernel<<<E,THREADS,0,stream.stream()>>>(
      s[0].data_ptr<int64_t>(), s[1].data_ptr<int64_t>(), s[2].data_ptr<int64_t>(),
      s[3].data_ptr<int64_t>(), s[4].data_ptr<bool>(), s[5].data_ptr<int64_t>(),
      s[6].data_ptr<int64_t>(), s[7].data_ptr<int64_t>(), s[8].data_ptr<int64_t>(),
      s[9].data_ptr<int64_t>(), s[10].data_ptr<bool>(), s[11].data_ptr<int64_t>(),
      s[12].data_ptr<int64_t>(), occupancy.data_ptr<int32_t>(), S,C,width,height);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> move_collisions_cuda(std::vector<torch::Tensor> s,
                                             bool v2, double ratio) {
  check(s, 9);
  const int64_t E = s[0].size(0), S = s[0].size(1), C = s[0].size(2);
  for (int a = 1; a <= 4; ++a) shape(s[a], a == 4 ? torch::kBool : torch::kInt64, {E,S});
  shape(s[5], torch::kInt64, {E,S,2}); shape(s[6], torch::kInt64, {E,S,2,2});
  shape(s[7], torch::kBool, {E,S,2}); shape(s[8], torch::kBool, {E,S});
  auto flags = torch::empty({E,S,S,2}, s[0].options().dtype(torch::kInt32));
  auto death = torch::empty({E,S}, s[0].options());
  auto ranks = torch::empty_like(death), kills = torch::empty_like(death), mass = torch::empty_like(death);
  auto dead = torch::empty({E,S}, s[0].options().dtype(torch::kBool));
  const c10::cuda::CUDAGuard guard(s[0].device());
  const auto stream = at::cuda::getCurrentCUDAStream(s[0].get_device());
  inspect_kernel<<<E*S*S,THREADS,0,stream.stream()>>>(
      s[0].data_ptr<int64_t>(), s[1].data_ptr<int64_t>(), s[2].data_ptr<int64_t>(),
      s[4].data_ptr<bool>(), s[5].data_ptr<int64_t>(), s[6].data_ptr<int64_t>(),
      s[7].data_ptr<bool>(), flags.data_ptr<int32_t>(), S,C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  resolve_kernel<<<(E+127)/128,128,0,stream.stream()>>>(
      s[3].data_ptr<int64_t>(), s[4].data_ptr<bool>(), s[8].data_ptr<bool>(),
      flags.data_ptr<int32_t>(), death.data_ptr<int64_t>(), ranks.data_ptr<int64_t>(),
      dead.data_ptr<bool>(), kills.data_ptr<int64_t>(), mass.data_ptr<int64_t>(), E,S,v2,float(ratio));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {death,ranks,dead,kills,mass};
}
