#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include "pi0/pi0.h"

namespace py = pybind11;

PYBIND11_MODULE(pi0_cpp, m) {
    m.doc() = "C++ bindings for Pi0 policy";

    py::class_<lerobot::pi0::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("n_obs_steps", &lerobot::pi0::Config::n_obs_steps)
        .def_readwrite("chunk_size", &lerobot::pi0::Config::chunk_size)
        .def_readwrite("n_action_steps", &lerobot::pi0::Config::n_action_steps);

    py::class_<lerobot::pi0::Pi0, std::shared_ptr<lerobot::pi0::Pi0>, torch::nn::Module>(m, "Pi0")
        .def(py::init<const lerobot::pi0::Config&>())
        .def("forward", &lerobot::pi0::Pi0::forward);

    m.def("create_pi0", &lerobot::pi0::create_pi0, py::arg("cfg"));
}

