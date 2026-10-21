#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flood_cuda(torch::Tensor start, torch::Tensor blocked,
                                     torch::Tensor cap, torch::Tensor release,
                                     int64_t max_cap);
std::vector<torch::Tensor> components_cuda(torch::Tensor allowed, torch::Tensor heads,
                                          torch::Tensor direction, torch::Tensor alive,
                                          torch::Tensor rows);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flood", &flood_cuda, "E1 sparse-frontier BFS with device watchdog (CUDA)");
  m.def("components", &components_cuda, "E1 tiled union-find with device watchdog (CUDA)");
}
