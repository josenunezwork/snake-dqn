#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <vector>

namespace {
constexpr int kThreads = 256;

// The CPU vector61 oracle evaluates its feature arithmetic as ordered IEEE-754
// binary64 operations and only then rounds each feature to float32.  CUDA's
// ordinary operators are permitted to contract or select target-specific
// implementations even when their source spelling looks equivalent.  Keep the
// scalar feature path explicit: each helper emits one correctly-rounded double
// operation and, in particular, never creates an FMA.  ``--fmad=false`` is a
// second (build-time) guard; these wrappers make that contract visible in the
// kernel itself.
// Project-specific names avoid CUDA headers' dadd/dsub/dmul overload aliases.
__device__ __forceinline__ double vector_rn_add(double lhs, double rhs) {
  return __dadd_rn(lhs, rhs);
}

__device__ __forceinline__ double vector_rn_sub(double lhs, double rhs) {
  return __dsub_rn(lhs, rhs);
}

__device__ __forceinline__ double vector_rn_mul(double lhs, double rhs) {
  return __dmul_rn(lhs, rhs);
}

__device__ __forceinline__ double vector_rn_div(double lhs, double rhs) {
  return __ddiv_rn(lhs, rhs);
}

__device__ __forceinline__ double vector_rn_sqrt(double value) {
  return __dsqrt_rn(value);
}

__device__ __forceinline__ double distance_from_squared(long long squared) {
  // All feature coordinates are integer pixels.  Form the squared distance in
  // integer arithmetic, exactly as the CPU reference does, before its one
  // correctly-rounded binary64 square root.
  return vector_rn_sqrt(static_cast<double>(squared));
}

__global__ void indexed_body_view_kernel(
    const int64_t* bodies, const int64_t* head_ptr, const int64_t* seg_count,
    const int64_t* length, const bool* alive, int64_t* cells, bool* live,
    bool* self_ok, bool* other_ok, int e_count, int slots, int cap) {
  const int64_t total = static_cast<int64_t>(e_count) * slots * cap;
  for (int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       flat < total; flat += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int k = static_cast<int>(flat % cap);
    const int64_t snake = flat / cap;
    const int ring = static_cast<int>((head_ptr[snake] - k + cap) % cap);
    const int64_t source = (snake * cap + ring) * 2;
    const int64_t dest = flat * 2;
    cells[dest] = bodies[source];
    cells[dest + 1] = bodies[source + 1];
    const int64_t n = seg_count[snake];
    const int64_t logical_length = length[snake];
    const bool is_live = alive[snake] && k < n;
    live[flat] = is_live;
    const bool drop_other = n > 1 && n >= logical_length;
    other_ok[flat] = is_live && k < n - static_cast<int>(drop_other);
    const bool drop_self = n >= logical_length && n > 2;
    self_ok[flat] = is_live && k >= 2 && k < n - static_cast<int>(drop_self);
  }
}

void check(const torch::Tensor& tensor, const torch::Device& device,
           torch::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.is_contiguous(),
              name, " must be contiguous CUDA on the shared device");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
}
}  // namespace

std::vector<torch::Tensor> indexed_body_view_cuda(torch::Tensor bodies, torch::Tensor head_ptr,
                                                   torch::Tensor seg_count, torch::Tensor length,
                                                   torch::Tensor alive) {
  c10::cuda::CUDAGuard guard(bodies.device());
  const auto device = bodies.device();
  check(bodies, device, torch::kLong, "bodies");
  check(head_ptr, device, torch::kLong, "head_ptr");
  check(seg_count, device, torch::kLong, "seg_count");
  check(length, device, torch::kLong, "length");
  check(alive, device, torch::kBool, "alive");
  TORCH_CHECK(bodies.dim() == 4 && bodies.size(3) == 2, "bodies must be (E,S,cap,2)");
  const auto e = bodies.size(0), s = bodies.size(1), cap = bodies.size(2);
  TORCH_CHECK(head_ptr.sizes() == torch::IntArrayRef({e, s}) &&
              seg_count.sizes() == head_ptr.sizes() && length.sizes() == head_ptr.sizes() &&
              alive.sizes() == head_ptr.sizes() && cap > 0, "ring-view shape mismatch");
  auto cells = torch::empty_like(bodies);
  auto flags = torch::empty({e, s, cap}, alive.options());
  auto self_ok = torch::empty_like(flags);
  auto other_ok = torch::empty_like(flags);
  const int64_t count = e * s * cap;
  if (!count) return {cells, flags, self_ok, other_ok};
  const auto stream = at::cuda::getCurrentCUDAStream(bodies.get_device());
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  indexed_body_view_kernel<<<blocks, kThreads, 0, stream.stream()>>>(
      bodies.data_ptr<int64_t>(), head_ptr.data_ptr<int64_t>(), seg_count.data_ptr<int64_t>(),
      length.data_ptr<int64_t>(), alive.data_ptr<bool>(), cells.data_ptr<int64_t>(),
      flags.data_ptr<bool>(), self_ok.data_ptr<bool>(), other_ok.data_ptr<bool>(),
      static_cast<int>(e), static_cast<int>(s), static_cast<int>(cap));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {cells, flags, self_ok, other_ok};
}

// One block owns one feature row.  Thread zero deliberately performs the
// reductions in slot/ring order: this trades some parallelism for the exact
// CPU contract's stable ties, duplicate multiplicity and double operation
// order.  Crucially it never builds row x action x snake x capacity tensors.
__global__ void indexed_features58_kernel(
    const int64_t* bodies, const int64_t* head_ptr, const int64_t* seg_count,
    const int64_t* length, const bool* alive, const int64_t* direction,
    const int64_t* food, const bool* food_present, const int64_t* rows,
    const double* prev_dist, const int64_t* prev_id, const int64_t* lut,
    float* out, double* next_dist, int64_t* next_id, int nr, int slots, int cap,
    int foods, int rx, int ry, int grid_w, int grid_h, int ss, int sectors,
    int max_length, int danger_distance, int min_boost, int food_capacity, bool boundary_danger) {
  if (threadIdx.x || blockIdx.x >= nr) return;
  const int r = blockIdx.x, env = static_cast<int>(rows[2 * r]), self = static_cast<int>(rows[2 * r + 1]);
  float* f = out + static_cast<int64_t>(r) * 58;
  for (int i = 0; i < 58; ++i) f[i] = 0.0F;
  const int64_t self_index = static_cast<int64_t>(env) * slots + self;
  if (!alive[self_index]) { next_dist[r] = prev_dist[self_index]; next_id[r] = prev_id[self_index]; return; }
  const int nself = static_cast<int>(seg_count[self_index]);
  const int self_ring = static_cast<int>((head_ptr[self_index] + cap) % cap);
  const int64_t self_base = (self_index * cap + self_ring) * 2;
  const int hx = static_cast<int>(bodies[self_base]) * ss, hy = static_cast<int>(bodies[self_base + 1]) * ss;
  const int d = static_cast<int>(direction[self_index]);
  const int cardx[4] = {0, 1, 0, -1}, cardy[4] = {-1, 0, 1, 0};
  f[d] = 1.0F;
  const int logical = max(1, static_cast<int>(length[self_index]));
  f[4] = __double2float_rn(fmin(1.0, vector_rn_div(static_cast<double>(logical),
                                            static_cast<double>(max_length))));
  const int max_dim = max(grid_w, grid_h) * ss;
  const long long width_px = static_cast<long long>(grid_w) * ss;
  const long long height_px = static_cast<long long>(grid_h) * ss;
  const double diag = distance_from_squared(width_px * width_px + height_px * height_px);
  int food_counts[16] = {};
  int best_food = -1; long long best_food_d2 = LLONG_MAX;
  for (int q = 0; q < foods; ++q) if (food_present[static_cast<int64_t>(env) * foods + q]) {
    const int dx = static_cast<int>(food[(static_cast<int64_t>(env) * foods + q) * 2]) * ss - hx;
    const int dy = static_cast<int>(food[(static_cast<int64_t>(env) * foods + q) * 2 + 1]) * ss - hy;
    const long long d2 = static_cast<long long>(dx) * dx + static_cast<long long>(dy) * dy;
    if (d2 < best_food_d2) { best_food_d2 = d2; best_food = q; }
    const int sec = static_cast<int>(lut[(dx / ss + rx) * (2 * ry + 1) + dy / ss + ry]);
    ++food_counts[sec];
  }
  // Mirrors Vector61Featurizer.expected_per_sector exactly: the denominator is
  // a separately rounded Python float division before the count division.
  const double expected_per_sector = fmax(
      1.0, vector_rn_div(static_cast<double>(food_capacity), static_cast<double>(sectors)));
  for (int sec = 0; sec < sectors; ++sec) {
    f[8 + sec] = __double2float_rn(fmin(
        1.0, vector_rn_div(static_cast<double>(food_counts[sec]), expected_per_sector)));
  }
  if (best_food >= 0) {
    const int dx = static_cast<int>(food[(static_cast<int64_t>(env) * foods + best_food) * 2]) * ss - hx;
    const int dy = static_cast<int>(food[(static_cast<int64_t>(env) * foods + best_food) * 2 + 1]) * ss - hy;
    f[5] = __double2float_rn(vector_rn_div(static_cast<double>(dx), static_cast<double>(max_dim)));
    f[6] = __double2float_rn(vector_rn_div(static_cast<double>(dy), static_cast<double>(max_dim)));
    f[7] = __double2float_rn(fmin(
        1.0, vector_rn_div(distance_from_squared(best_food_d2), diag)));
  } else f[7] = 1.0F;
  const double maxd = static_cast<double>(danger_distance * ss);
  double danger[16] = {}; int count[16] = {};
  double action_near[3] = {}; bool action_hit[3] = {};
  for (int a = 0; a < 3; ++a) {
    int nd = (d + a + 3) % 4, nx = hx + cardx[nd] * ss, ny = hy + cardy[nd] * ss;
    action_hit[a] = nx < 0 || nx >= grid_w * ss || ny < 0 || ny >= grid_h * ss;
    action_near[a] = fmin(3.0 * ss, fmin(fmin(nx, grid_w * ss - nx), fmin(ny, grid_h * ss - ny)));
  }
  for (int owner = 0; owner < slots; ++owner) {
    const int64_t oi = static_cast<int64_t>(env) * slots + owner;
    if (!alive[oi]) continue;
    const int n = static_cast<int>(seg_count[oi]), len = static_cast<int>(length[oi]);
    const int drop_other = n > 1 && n >= len, drop_self = n >= len && n > 2;
    for (int k = 0; k < n; ++k) {
      const bool use = owner == self ? (k >= 2 && k < n - drop_self) : (k < n - drop_other);
      if (!use) continue;
      const int ring = static_cast<int>((head_ptr[oi] - k + cap) % cap);
      const int64_t bi = (oi * cap + ring) * 2;
      const int bx = static_cast<int>(bodies[bi]) * ss, by = static_cast<int>(bodies[bi + 1]) * ss;
      const int dx = bx - hx, dy = by - hy;
      const long long d2 = static_cast<long long>(dx) * dx + static_cast<long long>(dy) * dy;
      const double dist = distance_from_squared(d2);
      if (dist <= maxd) {
        const int sec = static_cast<int>(lut[(dx / ss + rx) * (2 * ry + 1) + dy / ss + ry]);
        const double val = vector_rn_mul(fmax(0.0, vector_rn_sub(1.0, vector_rn_div(dist, maxd))), 1.5);
        danger[sec] = fmax(danger[sec], val);
        if (val > 0.0) ++count[sec];
      }
      for (int a = 0; a < 3; ++a) {
        int nd = (d + a + 3) % 4, nx = hx + cardx[nd] * ss, ny = hy + cardy[nd] * ss;
        const int adx = bx - nx, ady = by - ny;
        const double dd = distance_from_squared(
            static_cast<long long>(adx) * adx + static_cast<long long>(ady) * ady);
        if (dd < ss) action_hit[a] = true;
        action_near[a] = fmin(action_near[a], dd);
      }
    }
  }
  // Boundary samples follow the CPU's four ordered wall strips.  They are
  // points one segment outside the board, never implicit clamped distances.
  if (boundary_danger) for (int off = -danger_distance; off <= danger_distance; ++off) {
    const int delta = off * ss;
    const int wx[4] = {-ss, grid_w * ss + ss, hx + delta, hx + delta};
    const int wy[4] = {hy + delta, hy + delta, -ss, grid_h * ss + ss};
    const bool ok[4] = {
        hx < danger_distance * ss + ss && hy + delta >= 0 && hy + delta < grid_h * ss,
        grid_w * ss - hx < danger_distance * ss + ss && hy + delta >= 0 && hy + delta < grid_h * ss,
        hy < danger_distance * ss + ss && hx + delta >= 0 && hx + delta < grid_w * ss,
        grid_h * ss - hy < danger_distance * ss + ss && hx + delta >= 0 && hx + delta < grid_w * ss};
    for (int q = 0; q < 4; ++q) if (ok[q]) {
      const int dx = wx[q] - hx, dy = wy[q] - hy;
      const double dist = distance_from_squared(
          static_cast<long long>(dx) * dx + static_cast<long long>(dy) * dy);
      if (dist <= maxd) {
        const int sec = static_cast<int>(lut[(dx / ss + rx) * (2 * ry + 1) + dy / ss + ry]);
        const double val = vector_rn_mul(fmax(0.0, vector_rn_sub(1.0, vector_rn_div(dist, maxd))), 2.0);
        danger[sec] = fmax(danger[sec], val);
        if (val > 0.0) ++count[sec];
      }
    }
  }
  for (int sec = 0; sec < sectors; ++sec) {
    if (count[sec] > 1) {
      const double bonus = vector_rn_mul(fmin(1.0, vector_rn_div(static_cast<double>(count[sec]), 5.0)), .2);
      danger[sec] = fmin(2.0, vector_rn_add(danger[sec], bonus));
    }
    f[24 + sec] = __double2float_rn(fmin(1.0, danger[sec]));
  }
  f[40] = __double2float_rn(vector_rn_div(static_cast<double>(hx), static_cast<double>(width_px)));
  f[41] = __double2float_rn(vector_rn_div(static_cast<double>(width_px - hx), static_cast<double>(width_px)));
  f[42] = __double2float_rn(vector_rn_div(static_cast<double>(hy), static_cast<double>(height_px)));
  f[43] = __double2float_rn(vector_rn_div(static_cast<double>(height_px - hy), static_cast<double>(height_px)));
  int first = -1, second = -1; double firstd = INFINITY, secondd = INFINITY;
  for (int o = 0; o < slots; ++o) if (o != self && alive[static_cast<int64_t>(env)*slots+o]) {
    const int64_t oi = static_cast<int64_t>(env) * slots + o;
    const int ring = static_cast<int>((head_ptr[oi] + cap) % cap);
    const int64_t bi = (oi * cap + ring) * 2;
    const int dx = static_cast<int>(bodies[bi]) * ss - hx;
    const int dy = static_cast<int>(bodies[bi + 1]) * ss - hy;
    const double ed = distance_from_squared(
        static_cast<long long>(dx) * dx + static_cast<long long>(dy) * dy);
    if (ed < firstd) { second = first; secondd = firstd; first = o; firstd = ed; }
    else if (ed < secondd) { second = o; secondd = ed; }
  }
  next_dist[r]=INFINITY; next_id[r]=-1;
  if (first >= 0) {
    const int64_t oi = static_cast<int64_t>(env) * slots + first;
    const int ring = static_cast<int>((head_ptr[oi] + cap) % cap);
    const int64_t bi = (oi * cap + ring) * 2;
    const int ex = static_cast<int>(bodies[bi]) * ss, ey = static_cast<int>(bodies[bi + 1]) * ss;
    f[44] = __double2float_rn(vector_rn_div(static_cast<double>(ex - hx), static_cast<double>(max_dim)));
    f[45] = __double2float_rn(vector_rn_div(static_cast<double>(ey - hy), static_cast<double>(max_dim)));
    f[46] = __double2float_rn(vector_rn_div(fmin(2.0, vector_rn_div(static_cast<double>(max(1, static_cast<int>(length[oi]))),
                                           static_cast<double>(logical))), 2.0));
    f[47] = cardx[static_cast<int>(direction[oi])];
    f[48] = cardy[static_cast<int>(direction[oi])];
    if (prev_id[self_index] == first && isfinite(prev_dist[self_index])) {
      f[49] = firstd < prev_dist[self_index] ? 1.0F : firstd > prev_dist[self_index] ? -1.0F : 0.0F;
    }
    const int kdx = hx - (ex + cardx[static_cast<int>(direction[oi])] * ss);
    const int kdy = hy - (ey + cardy[static_cast<int>(direction[oi])] * ss);
    f[53] = distance_from_squared(static_cast<long long>(kdx) * kdx +
                                  static_cast<long long>(kdy) * kdy) < 3 * ss;
    next_dist[r] = firstd;
    next_id[r] = first;
  }
  if (second >= 0) {
    const int64_t oi = static_cast<int64_t>(env) * slots + second;
    const int ring = (head_ptr[oi] + cap) % cap;
    const int64_t bi = (oi * cap + ring) * 2;
    f[50] = __double2float_rn(vector_rn_div(static_cast<double>(static_cast<int>(bodies[bi]) * ss - hx),
                                   static_cast<double>(max_dim)));
    f[51] = __double2float_rn(vector_rn_div(static_cast<double>(static_cast<int>(bodies[bi + 1]) * ss - hy),
                                   static_cast<double>(max_dim)));
    f[52] = __double2float_rn(vector_rn_div(fmin(2.0, vector_rn_div(static_cast<double>(max(1, static_cast<int>(length[oi]))),
                                           static_cast<double>(logical))), 2.0));
  }
  for (int a = 0; a < 3; ++a) {
    f[54 + a] = action_hit[a] ? 1.0F : __double2float_rn(
        action_near[a] < 3 * ss ? fmin(.95, vector_rn_sub(1.0, vector_rn_div(action_near[a], 3.0 * ss))) : 0.0);
  }
  f[57]=length[self_index]>=min_boost;
}

std::vector<torch::Tensor> indexed_features58_cuda(torch::Tensor bodies, torch::Tensor head_ptr, torch::Tensor seg_count, torch::Tensor length, torch::Tensor alive, torch::Tensor direction, torch::Tensor food, torch::Tensor food_present, torch::Tensor rows, torch::Tensor prev_dist, torch::Tensor prev_id, torch::Tensor sector_lut, int64_t rx, int64_t ry, int64_t grid_w, int64_t grid_h, int64_t segment_size, int64_t sectors, int64_t max_length, int64_t danger_distance, int64_t min_boost, int64_t food_capacity, bool boundary_danger) {
  c10::cuda::CUDAGuard guard(bodies.device()); auto device=bodies.device(); check(bodies,device,torch::kLong,"bodies"); check(rows,device,torch::kLong,"rows"); check(prev_dist,device,torch::kDouble,"prev_dist"); check(prev_id,device,torch::kLong,"prev_id"); check(sector_lut,device,torch::kLong,"sector_lut");
  check(head_ptr,device,torch::kLong,"head_ptr"); check(seg_count,device,torch::kLong,"seg_count"); check(length,device,torch::kLong,"length"); check(alive,device,torch::kBool,"alive"); check(direction,device,torch::kLong,"direction"); check(food,device,torch::kLong,"food"); check(food_present,device,torch::kBool,"food_present");
  const int e=bodies.size(0),s=bodies.size(1),cap=bodies.size(2),nr=rows.size(0),foods=food.size(1); TORCH_CHECK(bodies.dim()==4&&bodies.size(3)==2&&head_ptr.sizes()==torch::IntArrayRef({e,s})&&seg_count.sizes()==head_ptr.sizes()&&length.sizes()==head_ptr.sizes()&&alive.sizes()==head_ptr.sizes()&&direction.sizes()==head_ptr.sizes()&&food.dim()==3&&food.size(0)==e&&food.size(2)==2&&food_present.sizes()==torch::IntArrayRef({e,foods})&&rows.dim()==2&&rows.size(1)==2&&prev_dist.sizes()==head_ptr.sizes()&&prev_id.sizes()==head_ptr.sizes(),"indexed feature tensor shape mismatch"); auto out=torch::zeros({nr,58},bodies.options().dtype(torch::kFloat)); auto nd=torch::empty({nr},prev_dist.options()); auto ni=torch::empty({nr},prev_id.options()); if(!nr)return {out,nd,ni}; auto stream=at::cuda::getCurrentCUDAStream(bodies.get_device()); indexed_features58_kernel<<<nr,1,0,stream.stream()>>>(bodies.data_ptr<int64_t>(),head_ptr.data_ptr<int64_t>(),seg_count.data_ptr<int64_t>(),length.data_ptr<int64_t>(),alive.data_ptr<bool>(),direction.data_ptr<int64_t>(),food.data_ptr<int64_t>(),food_present.data_ptr<bool>(),rows.data_ptr<int64_t>(),prev_dist.data_ptr<double>(),prev_id.data_ptr<int64_t>(),sector_lut.data_ptr<int64_t>(),out.data_ptr<float>(),nd.data_ptr<double>(),ni.data_ptr<int64_t>(),nr,s,cap,foods,rx,ry,grid_w,grid_h,segment_size,sectors,max_length,danger_distance,min_boost,food_capacity,boundary_danger); C10_CUDA_KERNEL_LAUNCH_CHECK(); return {out,nd,ni};
}

