#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> v8_indexed_construct_cuda(
    torch::Tensor bodies, torch::Tensor head_ptr, torch::Tensor seg_count,
    torch::Tensor length, torch::Tensor alive, torch::Tensor direction,
    torch::Tensor boost_frames, torch::Tensor rows, int64_t h, int64_t w,
    int64_t min_boost_length, int64_t boost_length_cost_frames);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("construct", &v8_indexed_construct_cuda,
        "V8 direct-ring occupancy/post-body/release construction (CUDA)");
}
