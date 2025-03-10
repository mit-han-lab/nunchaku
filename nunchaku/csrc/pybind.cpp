#include "gemm.h"
#include "gemm88.h"
#include "flux.h"
#include "sana.h"
#include "ops.h"
#include "utils.h"
#include <torch/extension.h>

#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<QuantizedFluxModel>(m, "QuantizedFluxModel")
        .def(py::init<>())
        .def("init", &QuantizedFluxModel::init,
            py::arg("use_fp4"),
            py::arg("offload"),
            py::arg("bf16"),
            py::arg("deviceId")
        )
        .def("reset", &QuantizedFluxModel::reset)
        .def("load", &QuantizedFluxModel::load, 
            py::arg("path"),
            py::arg("partial") = false
        )
        .def("loadDict", &QuantizedFluxModel::loadDict, 
            py::arg("dict"),
            py::arg("partial") = false
        )
        .def("forward", &QuantizedFluxModel::forward)
        .def("forward_layer", &QuantizedFluxModel::forward_layer)
        .def("forward_single_layer", &QuantizedFluxModel::forward_single_layer)
        .def("startDebug", &QuantizedFluxModel::startDebug)
        .def("stopDebug", &QuantizedFluxModel::stopDebug)
        .def("getDebugResults", &QuantizedFluxModel::getDebugResults)
        .def("setLoraScale", &QuantizedFluxModel::setLoraScale)
        .def("forceFP16Attention", &QuantizedFluxModel::forceFP16Attention)
    ;
    py::class_<QuantizedSanaModel>(m, "QuantizedSanaModel")
        .def(py::init<>())
        .def("init", &QuantizedSanaModel::init,
            py::arg("config"),
            py::arg("pag_layers"),
            py::arg("use_fp4"),
            py::arg("bf16"),
            py::arg("deviceId")
        )
        .def("reset", &QuantizedSanaModel::reset)
        .def("load", &QuantizedSanaModel::load, 
            py::arg("path"),
            py::arg("partial") = false
        )
        .def("loadDict", &QuantizedSanaModel::loadDict, 
            py::arg("dict"),
            py::arg("partial") = false
        )
        .def("forward", &QuantizedSanaModel::forward)
        .def("forward_layer", &QuantizedSanaModel::forward_layer)
        .def("startDebug", &QuantizedSanaModel::startDebug)
        .def("stopDebug", &QuantizedSanaModel::stopDebug)
        .def("getDebugResults", &QuantizedSanaModel::getDebugResults)
    ;
    py::class_<QuantizedGEMM>(m, "QuantizedGEMM")
        .def(py::init<>())
        .def("init", &QuantizedGEMM::init)
        .def("reset", &QuantizedGEMM::reset)
        .def("load", &QuantizedGEMM::load)
        .def("forward", &QuantizedGEMM::forward)
        .def("quantize", &QuantizedGEMM::quantize)
        .def("startDebug", &QuantizedGEMM::startDebug)
        .def("stopDebug", &QuantizedGEMM::stopDebug)
        .def("getDebugResults", &QuantizedGEMM::getDebugResults)
    ;
    py::class_<QuantizedGEMM88>(m, "QuantizedGEMM88")
        .def(py::init<>())
        .def("init", &QuantizedGEMM88::init)
        .def("reset", &QuantizedGEMM88::reset)
        .def("load", &QuantizedGEMM88::load)
        .def("forward", &QuantizedGEMM88::forward)
        .def("startDebug", &QuantizedGEMM88::startDebug)
        .def("stopDebug", &QuantizedGEMM88::stopDebug)
        .def("getDebugResults", &QuantizedGEMM88::getDebugResults)
    ;

    m.def_submodule("ops")
        .def("gemm_awq", nunchaku::ops::gemm_awq)
        .def("gemv_awq", nunchaku::ops::gemv_awq)
    ;

    m.def_submodule("utils")
        .def("set_log_level", [](const std::string &level) {
            spdlog::set_level(spdlog::level::from_str(level));
        })
        .def("disable_memory_auto_release", nunchaku::utils::disable_memory_auto_release)
        .def("trim_memory", nunchaku::utils::trim_memory)
    ;
}
