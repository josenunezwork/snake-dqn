#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <vector>

namespace {
constexpr int kTile = 16;
constexpr int kThreads = kTile * kTile;

// Parents decrease strictly at every successful union. Atomic root reads and
// CAS root links make cycles impossible even when boundary edges race.
__device__ int root_of(int* parent, int cell, int vertices, int* status) {
  for (int step = 0; step < vertices; ++step) {
    int next = atomicAdd(parent + cell, 0);
    if (next == cell) return cell;
    if (next < 0 || next >= vertices || next > cell) {
      atomicOr(status, 1);
      return -1;
    }
    int grand = atomicAdd(parent + next, 0);
    if (grand < 0 || grand > next) {
      atomicOr(status, 1);
      return -1;
    }
    atomicMin(parent + cell, grand);  // pointer jumping, never a larger parent
    cell = next;
  }
  atomicOr(status, 1);  // visible convergence watchdog, never partial success
  return -1;
}

__device__ void join_roots(int* parent, int a, int b, int vertices, int* status) {
  for (int retry = 0; retry < vertices; ++retry) {
    a = root_of(parent, a, vertices, status);
    b = root_of(parent, b, vertices, status);
    if (a < 0 || b < 0 || a == b) return;
    int hi = max(a, b), lo = min(a, b);
    if (atomicCAS(parent + hi, hi, lo) == hi) return;
  }
  atomicOr(status, 2);
}

// Each independent tile labels its true local components before any boundary
// work. Shared scratch is 1 KiB, independent of the arena or world count.
__global__ void local_components(const bool* allowed, int* parent, int* status,
                                 int vertices, int h, int w, int tiles_x, int tiles_y) {
  __shared__ int local[kThreads];
  int tile_count = tiles_x * tiles_y;
  int env = blockIdx.x / tile_count, tile = blockIdx.x % tile_count;
  int tx = tile % tiles_x, ty = tile / tiles_x;
  int tid = threadIdx.x, lx = tid % kTile, ly = tid / kTile;
  int x = tx * kTile + lx, y = ty * kTile + ly;
  bool open = x < w && y < h && allowed[static_cast<int64_t>(env) * vertices + y * w + x];
  local[tid] = open ? tid : -1;
  __syncthreads();
  if (open) {
    if (lx + 1 < kTile && atomicAdd(local + tid + 1, 0) >= 0)
      join_roots(local, tid, tid + 1, kThreads, status);
    if (ly + 1 < kTile && atomicAdd(local + tid + kTile, 0) >= 0)
      join_roots(local, tid, tid + kTile, kThreads, status);
  }
  __syncthreads();
  if (x < w && y < h) {
    int root = open ? root_of(local, tid, kThreads, status) : -1;
    int global_root = root < 0 ? -1 : (ty * kTile + root / kTile) * w + tx * kTile + root % kTile;
    parent[static_cast<int64_t>(env) * vertices + y * w + x] = global_root;
  }
}

// The preceding launch is the grid barrier. Only edges crossing tile borders
// remain; CAS union follows roots across any number of already joined tiles.
__global__ void boundary_components(const bool* allowed, int* parent, int* status,
                                    int64_t total, int vertices, int h, int w) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total || !allowed[index]) return;
  int cell = index % vertices, x = cell % w, y = cell / w;
  int* world_parent = parent + (index / vertices) * vertices;
  if (x % kTile == kTile - 1 && x + 1 < w && allowed[index + 1])
    join_roots(world_parent, cell, cell + 1, vertices, status);
  if (y % kTile == kTile - 1 && y + 1 < h && allowed[index + w])
    join_roots(world_parent, cell, cell + w, vertices, status);
}

// A second grid barrier fixes every union. Chase/jump to the stable root and
// reduce integer sizes once per WORLD, independently of requested row count.
__global__ void count_components(const bool* allowed, int* parent, int* counts,
                                 int* status, int64_t total, int vertices) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total || !allowed[index]) return;
  int64_t offset = (index / vertices) * vertices;
  int root = root_of(parent + offset, index % vertices, vertices, status);
  if (root >= 0) {
    atomicMin(parent + index, root);
    atomicAdd(counts + offset + root, 1);
  }
}

__global__ void gather_components(const int* parent, const int* counts,
                                  const int64_t* heads, const int64_t* directions,
                                  const bool* alive, const int64_t* rows, int64_t* out,
                                  int* status, int nr, int worlds, int slots, int h, int w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * 3) return;
  int row = i / 3, rel = i % 3;
  int64_t env = rows[row * 2], slot = rows[row * 2 + 1];
  if (env < 0 || env >= worlds || slot < 0 || slot >= slots) {
    atomicOr(status, 4);
    return;
  }
  int64_t snake = env * slots + slot;
  if (!alive[snake]) return;
  int d = directions[snake];
  if (d < 0 || d > 3) { atomicOr(status, 4); return; }
  int dx[4] = {0, 1, 0, -1}, dy[4] = {-1, 0, 1, 0};
  d = (d + rel + 3) % 4;
  int64_t x = heads[snake * 2] + dx[d], y = heads[snake * 2 + 1] + dy[d];
  if (x < 0 || x >= w || y < 0 || y >= h) return;
  int64_t offset = env * h * w;
  int root = parent[offset + y * w + x];
  if (root >= 0) out[i] = counts[offset + root];
}

// One block per world/slot/action. The queue is sparse and global; the seen
// bitset is small shared scratch (1508 bytes at 145x83). A batch never mixes
// distances: release-rejected cells stay unseen and may be reached later from
// a different accepted frontier, exactly as in the independent FIFO oracle.
__global__ void sparse_frontier(const int64_t* start, const bool* blocked,
                                const int64_t* cap, const int64_t* release,
                                int* queue, int* distance, int64_t* out, int* status,
                                int lanes, int h, int w, int qcapacity, int max_cap) {
  int lane = blockIdx.x;
  if (lane >= lanes) return;
  int vertices = h * w;
  extern __shared__ unsigned int seen[];
  for (int i = threadIdx.x; i < (vertices + 31) / 32; i += blockDim.x) seen[i] = 0;
  __syncthreads();  // all clears must precede source insertion
  __shared__ int begin, end, tail, popped, batch_begin, batch_count, done, wanted;
  int64_t offset = static_cast<int64_t>(lane) * vertices;
  int* cells = queue + static_cast<int64_t>(lane) * qcapacity;
  int* depths = distance + static_cast<int64_t>(lane) * qcapacity;
  if (threadIdx.x == 0) {
    begin = end = tail = popped = done = 0;
    int64_t raw_cap = cap[lane];
    wanted = raw_cap < vertices ? static_cast<int>(raw_cap) : vertices;
    if (raw_cap <= 0 || wanted > max_cap) {
      atomicOr(status, 8);
      done = 1;
    }
    int64_t x = start[lane * 2], y = start[lane * 2 + 1];
    if (!done && x >= 0 && x < w && y >= 0 && y < h) {
      int cell = y * w + x;
      if (!blocked[offset + cell] && (!release || release[offset + cell] <= 0)) {
        seen[cell >> 5] |= 1u << (cell & 31);
        cells[0] = cell;
        depths[0] = 0;
        end = tail = 1;
      }
    }
  }
  __syncthreads();
  for (int iteration = 0; iteration <= vertices; ++iteration) {
    if (threadIdx.x == 0) {
      done |= begin >= end || popped >= wanted;
      batch_begin = begin;
      batch_count = done ? 0 : min(static_cast<int>(blockDim.x), min(end - begin, wanted - popped));
      begin += batch_count;
      popped += batch_count;
    }
    __syncthreads();
    if (done) break;
    if (threadIdx.x < batch_count) {
      int pos = batch_begin + threadIdx.x, cell = cells[pos], d = depths[pos];
      int x = cell % w, y = cell / w;
      int nb[4] = {cell - w, cell + w, cell - 1, cell + 1};
      bool valid[4] = {y > 0, y + 1 < h, x > 0, x + 1 < w};
      for (int k = 0; k < 4; ++k) {
        int next = nb[k];
        if (!valid[k] || blocked[offset + next] ||
            (release && release[offset + next] > d + 1)) continue;
        unsigned int bit = 1u << (next & 31);
        if ((atomicOr(seen + (next >> 5), bit) & bit) == 0) {
          int at = atomicAdd(&tail, 1);
          if (at < qcapacity) { cells[at] = next; depths[at] = d + 1; }
          else atomicOr(status, 16);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      // The proven queue bound makes this unreachable for valid inputs, but
      // even a watchdog failure must not feed an out-of-range queue read.
      if (tail > qcapacity) done = 1;
      if (begin == end) end = min(tail, qcapacity);
    }
    __syncthreads();
    if (iteration == vertices && threadIdx.x == 0) atomicOr(status, 32);
  }
  if (threadIdx.x == 0) out[lane] = popped;
}

void check_tensor(const torch::Tensor& tensor, const torch::Device& device,
                  torch::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device, name, " must share the CUDA device");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.is_contiguous(), name, " dtype/contiguity mismatch");
}
}  // namespace

std::vector<torch::Tensor> flood_cuda(torch::Tensor start, torch::Tensor blocked,
                                     torch::Tensor cap, torch::Tensor release,
                                     int64_t max_cap) {
  c10::cuda::CUDAGuard guard(start.device());
  auto device = start.device();
  check_tensor(start, device, torch::kLong, "start");
  check_tensor(blocked, device, torch::kBool, "blocked");
  check_tensor(cap, device, torch::kLong, "cap");
  check_tensor(release, device, torch::kLong, "release");
  TORCH_CHECK(start.dim() == 3 && start.size(2) == 2 && blocked.dim() == 4, "invalid flood shapes");
  auto n = start.size(0), a = start.size(1), h = blocked.size(2), w = blocked.size(3);
  TORCH_CHECK(blocked.size(0) == n && blocked.size(1) == a && cap.dim() == 2 &&
              cap.size(0) == n && cap.size(1) == a, "flood lane mismatch");
  TORCH_CHECK(release.numel() == 0 || release.sizes() == blocked.sizes(), "release shape mismatch");
  TORCH_CHECK(h > 0 && w > 0 && h * w <= INT_MAX && n * a <= INT_MAX &&
              max_cap > 0 && max_cap <= 4096, "unsupported flood dimensions/cap");
  int64_t vertices = h * w, lanes = n * a;
  int qcapacity = std::min<int64_t>(vertices, 1 + 4 * max_cap);
  auto out = torch::zeros({n, a}, start.options());
  auto status = torch::zeros({}, start.options().dtype(torch::kInt32));
  if (lanes == 0) return {out, status};
  auto queue = torch::empty({lanes, qcapacity}, start.options().dtype(torch::kInt32));
  auto distance = torch::empty_like(queue);
  auto stream = at::cuda::getCurrentCUDAStream(start.get_device());
  sparse_frontier<<<lanes, kThreads, ((vertices + 31) / 32) * sizeof(unsigned int), stream.stream()>>>(
      start.data_ptr<int64_t>(), blocked.data_ptr<bool>(), cap.data_ptr<int64_t>(),
      release.numel() ? release.data_ptr<int64_t>() : nullptr, queue.data_ptr<int>(),
      distance.data_ptr<int>(), out.data_ptr<int64_t>(), status.data_ptr<int>(),
      lanes, h, w, qcapacity, max_cap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, status};
}

std::vector<torch::Tensor> components_cuda(torch::Tensor allowed, torch::Tensor heads,
                                          torch::Tensor direction, torch::Tensor alive,
                                          torch::Tensor rows) {
  c10::cuda::CUDAGuard guard(allowed.device());
  auto device = allowed.device();
  check_tensor(allowed, device, torch::kBool, "allowed");
  check_tensor(heads, device, torch::kLong, "heads");
  check_tensor(direction, device, torch::kLong, "direction");
  check_tensor(alive, device, torch::kBool, "alive");
  check_tensor(rows, device, torch::kLong, "rows");
  TORCH_CHECK(allowed.dim() == 3 && heads.dim() == 3 && heads.size(2) == 2 &&
              direction.dim() == 2 && rows.dim() == 2 && rows.size(1) == 2, "invalid component shapes");
  auto e = allowed.size(0), h = allowed.size(1), w = allowed.size(2), s = heads.size(1), nr = rows.size(0);
  TORCH_CHECK(heads.size(0) == e && direction.size(0) == e && direction.size(1) == s &&
              alive.sizes() == direction.sizes(), "component world/slot mismatch");
  TORCH_CHECK(h > 0 && w > 0 && h * w <= INT_MAX && nr * 3 <= INT_MAX &&
              e * h * w <= INT_MAX, "unsupported component dimensions");
  auto out = torch::zeros({nr, 3}, rows.options());
  auto status = torch::zeros({}, rows.options().dtype(torch::kInt32));
  if (nr == 0 || e == 0) return {out, status};
  int64_t total = e * h * w, vertices = h * w;
  auto parent = torch::empty({e, vertices}, rows.options().dtype(torch::kInt32));
  auto counts = torch::zeros_like(parent);
  int tiles_x = (w + kTile - 1) / kTile, tiles_y = (h + kTile - 1) / kTile;
  auto stream = at::cuda::getCurrentCUDAStream(allowed.get_device());
  local_components<<<e * tiles_x * tiles_y, kThreads, 0, stream.stream()>>>(
      allowed.data_ptr<bool>(), parent.data_ptr<int>(), status.data_ptr<int>(), vertices, h, w, tiles_x, tiles_y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  boundary_components<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream.stream()>>>(
      allowed.data_ptr<bool>(), parent.data_ptr<int>(), status.data_ptr<int>(), total, vertices, h, w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  count_components<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream.stream()>>>(
      allowed.data_ptr<bool>(), parent.data_ptr<int>(), counts.data_ptr<int>(), status.data_ptr<int>(), total, vertices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gather_components<<<(nr * 3 + kThreads - 1) / kThreads, kThreads, 0, stream.stream()>>>(
      parent.data_ptr<int>(), counts.data_ptr<int>(), heads.data_ptr<int64_t>(), direction.data_ptr<int64_t>(),
      alive.data_ptr<bool>(), rows.data_ptr<int64_t>(), out.data_ptr<int64_t>(), status.data_ptr<int>(),
      nr, e, s, h, w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, status};
}
