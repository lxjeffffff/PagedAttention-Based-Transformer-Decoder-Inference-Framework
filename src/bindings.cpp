#include "../include/bindings.hpp"

void register_decoder(py::module& m) {
    // CUDA float decoder
    py::class_<CUDADecoder<float>>(m, "CUDADecoder")
        .def(py::init<int, int, int, int, int, int>())
        .def("load_weights", &CUDADecoder<float>::load_weights)
        .def("generate", [](CUDADecoder<float>& self,
                            const std::vector<int>& input_ids,
                            int max_len,
                            float temperature) {
            std::vector<int> output_ids;
            self.generate(input_ids, output_ids, max_len, temperature);
            return output_ids;
        });

    // INT8 decoder
    py::class_<INT8Decoder>(m, "INT8Decoder")
        .def(py::init<int, int, int, int, int, int>())
        .def("load_quantized_weights", &INT8Decoder::load_quantized_weights)
        .def("quantize_weights", &INT8Decoder::quantize_weights)
        .def("generate", [](INT8Decoder& self,
                            const std::vector<int>& input_ids,
                            int max_len,
                            float temperature) {
            std::vector<int> output_ids;
            self.generate(input_ids, output_ids, max_len, temperature);
            return output_ids;
        });
}

PYBIND11_MODULE(llm_decoder, m) {
    m.doc() = "Python binding for CUDA/INT8 Transformer decoder";
    register_decoder(m);
}
