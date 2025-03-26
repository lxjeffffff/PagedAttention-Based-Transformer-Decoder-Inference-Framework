from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "llm_decoder",
        [
            "src/bindings.cpp",
            "decoder/cuda_decoder.cu",
            "decoder/int8_decoder.cpp",
            "decoder/decoder_block.hpp",
            "decoder/mlp.hpp",
            "decoder/layer_norm.hpp",
            "decoder/token_embedding.hpp",
            "attention_cpu/cpu_attention_kernel.cpp",
            "attention_cpu/softmax_lut.cpp",
            "attention_cpu/int8_quant.cpp",
            "attention_cpu/dnnl_matmul_int8.cpp",
            "kv_cache/kv_tile_cache.cpp",
            "kv_cache/kv_tile_cache_cpu.cpp",
            "kv_cache/page_table.cpp"
        ],
        include_dirs=[
            "include",
            "decoder",
            "attention",
            "attention_cpu",
            "kv_cache"
        ],
        extra_compile_args=["-O3"],
        cxx_std=17,
    )
]

setup(
    name="llm_decoder",
    version="0.1.0",
    author="Your Name",
    description="C++/CUDA + INT8 Transformer decoder module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
