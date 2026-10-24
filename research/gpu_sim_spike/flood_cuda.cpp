#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flood_cuda(torch::Tensor start, torch::Tensor blocked,
                                     torch::Tensor cap, torch::Tensor release,
                                     int64_t max_cap);
std::vector<torch::Tensor> components_cuda(torch::Tensor allowed, torch::Tensor heads,
                                          torch::Tensor direction, torch::Tensor alive,
                                          torch::Tensor rows);
std::vector<torch::Tensor> static_capped_cuda(torch::Tensor start, torch::Tensor occupied,
                                              torch::Tensor cap);
void set_launch_diagnostics(bool enabled, bool reset, bool capture_events);
pybind11::list get_launch_diagnostics();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flood", &flood_cuda, "E1 sparse-frontier BFS with device watchdog (CUDA)");
  m.def("components", &components_cuda, "E1 tiled union-find with device watchdog (CUDA)");
  m.def("static_capped", &static_capped_cuda,
        "E2 indexed warp-cooperative static capped BFS (CUDA)");
  m.def("set_launch_diagnostics", &set_launch_diagnostics, pybind11::arg("enabled"),
        pybind11::arg("reset") = false, pybind11::arg("capture_events") = true);
  m.def("get_launch_diagnostics", &get_launch_diagnostics);
}
