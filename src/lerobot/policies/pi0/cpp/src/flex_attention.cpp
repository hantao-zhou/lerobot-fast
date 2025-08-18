#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Declaration of the CUDA implementation.
torch::Tensor flex_attention_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask,
    double scale);

struct FlexAttention {
    static torch::Tensor forward(
        torch::Tensor query,
        torch::Tensor key,
        torch::Tensor value,
        torch::Tensor mask,
        double scale) {
        return flex_attention_forward_cuda(query, key, value, mask, scale);
    }
};

PYBIND11_MODULE(flex_attention_cpp, m) {
    py::class_<FlexAttention>(m, "FlexAttention")
        .def_static("forward", &FlexAttention::forward,
                    py::arg("query"),
                    py::arg("key"),
                    py::arg("value"),
                    py::arg("mask"),
                    py::arg("scale"));
}
