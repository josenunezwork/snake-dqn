#include <torch/extension.h>
void move_respawn_cuda(std::vector<torch::Tensor> state, torch::Tensor occupancy,
                       int64_t width, int64_t height);
std::vector<torch::Tensor> move_collisions_cuda(std::vector<torch::Tensor> state,
                                             bool v2, double ratio);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("respawn", &move_respawn_cuda, "MOVE single-build ordered respawn");
  m.def("collisions", &move_collisions_cuda, "MOVE indexed collision reduction");
}
