#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> indexed_body_view_cuda(torch::Tensor bodies, torch::Tensor head_ptr,
                                                   torch::Tensor seg_count, torch::Tensor length,
                                                   torch::Tensor alive);
std::vector<torch::Tensor> indexed_features58_cuda(
    torch::Tensor bodies, torch::Tensor head_ptr, torch::Tensor seg_count, torch::Tensor length,
    torch::Tensor alive, torch::Tensor direction, torch::Tensor food, torch::Tensor food_present,
    torch::Tensor rows, torch::Tensor prev_dist, torch::Tensor prev_id, torch::Tensor sector_lut,
    int64_t rx, int64_t ry, int64_t grid_w, int64_t grid_h, int64_t segment_size,
    int64_t sectors, int64_t max_length, int64_t danger_distance, int64_t min_boost,
    int64_t food_capacity, bool boundary_danger);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexed_body_view", &indexed_body_view_cuda,
        "Indexed ring-order vector61 body view (CUDA)");
  m.def("indexed_features58", &indexed_features58_cuda,
        "Direct indexed vector61 non-flood features (CUDA)");
}
