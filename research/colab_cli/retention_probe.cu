#include <torch/extension.h>
// A fixed compile/load proof with no mutable scientific state or reduction.
torch::Tensor identity(torch::Tensor value) { return value.clone(); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("identity", &identity); }
