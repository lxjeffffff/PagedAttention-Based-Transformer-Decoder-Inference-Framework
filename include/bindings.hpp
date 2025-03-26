#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../decoder/cuda_decoder.hpp"
#include "../decoder/int8_decoder.hpp"

namespace py = pybind11;

void register_decoder(py::module& m);
