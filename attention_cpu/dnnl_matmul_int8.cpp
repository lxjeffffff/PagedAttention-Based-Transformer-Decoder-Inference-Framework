#include "dnnl_matmul_int8.hpp"
#include <dnnl.hpp>
#include <unordered_map>

using namespace dnnl;

bool dnnl_matmul_int8(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int BATCH, int M, int N, int K,
    float scaleA, float scaleB, float scaleC,
    const float* bias,
    const std::string& activation)
{
    try {
        engine eng(engine::kind::cpu, 0);
        stream s(eng);

        memory::dims a_dims = {BATCH, M, K};
        memory::dims b_dims = {BATCH, K, N};
        memory::dims c_dims = {BATCH, M, N};
        memory::dims bias_dims = {1, 1, N};  // broadcast bias

        auto a_md = memory::desc(a_dims, memory::data_type::s8, memory::format_tag::abc);
        auto b_md = memory::desc(b_dims, memory::data_type::s8, memory::format_tag::abc);
        auto c_md = memory::desc(c_dims, memory::data_type::s8, memory::format_tag::abc);

        memory a_mem(a_md, eng, (void*)A);
        memory b_mem(b_md, eng, (void*)B);
        memory c_mem(c_md, eng, (void*)C);

        memory bias_mem;
        if (bias) {
            auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::abc);
            bias_mem = memory(bias_md, eng, (void*)bias);
        }

        primitive_attr attr;
        float alpha = scaleA * scaleB / scaleC;
        attr.set_output_scales(0, {alpha});

        post_ops ops;
        if (!activation.empty()) {
            if (activation == "relu") {
                ops.append_eltwise(1.0f, algorithm::eltwise_relu, 0.f, 0.f);
            } else if (activation == "gelu") {
                ops.append_eltwise(1.0f, algorithm::eltwise_gelu_erf, 0.f, 0.f);
            }
        }

        attr.set_post_ops(ops);

        auto matmul_d = bias
            ? matmul::desc(a_md, b_md, bias_mem.get_desc(), c_md)
            : matmul::desc(a_md, b_md, c_md);

        auto matmul_pd = matmul::primitive_desc(matmul_d, attr, eng);

        std::unordered_map<int, memory> args = {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        };
        if (bias) {
            args[DNNL_ARG_BIAS] = bias_mem;
        }

        matmul(matmul_pd).execute(s, args);
        s.wait();

        return true;
    } catch (...) {
        return false;
    }
}
