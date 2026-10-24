#include <torch/extension.h>
void append_fifo_cuda(torch::Tensor tag, torch::Tensor queue, torch::Tensor count,
                      torch::Tensor cells, torch::Tensor mask, torch::Tensor accepted,
                      int64_t old_food, int64_t cap, bool corpse, bool diagnostics);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("append_fifo", &append_fifo_cuda, "A0 ordered compact corpse FIFO");
}
