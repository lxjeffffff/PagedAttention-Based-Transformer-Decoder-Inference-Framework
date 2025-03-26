# `attention_cpu/` — High-Performance CPU Attention Module (float & int8)

This module implements a CPU-optimized Paged Attention kernel designed for transformer decoders. It supports both `float32` and `int8` inference paths, tile-based KV caching, rotary embeddings, top-k/p sampling, and fast vectorized operations using AVX2/NEON and LUT-based softmax.

---

### 📦 Contents

| File                              | Description |
|-----------------------------------|-------------|
| `attention_cpu.hpp/cpp`           | `CPUAttention<T>` wrapper for float/int8 inference |
| `cpu_attention_config.hpp`        | Config struct (head_dim, tile_size, etc.) |
| `cpu_attention_kernel.hpp/cpp`    | Core `cpu_paged_attention_forward<T>()` kernel |
| `attention_kernel_utils_cpu.hpp`  | CPU kernel helpers (masking, rotary, causal, beam EOS) |
| `dnnl_matmul_int8.hpp/cpp`        | INT8 matmul (GEMM) wrapper using oneDNN |
| `int8_quant.hpp/cpp`              | Quantization tools: FP32 → INT8, min/max scaling |
| `softmax_lut.hpp/cpp`             | LUT-accelerated softmax for INT8 inference |
| `vec_cpu.hpp`                     | CPU vector intrinsics abstraction (float/int8 support) |

---

### 🚀 Features

- ✅ **Paged Attention** with tile-based KV caching
- ✅ **Rotary Position Embedding (RoPE)**
- ✅ **Causal Masking**
- ✅ **Top-k / Top-p filtering**
- ✅ **EOS token detection**
- ✅ **INT8 inference** with:
  - oneDNN int8 GEMM
  - LUT-based softmax
  - vectorized INT8 arithmetic
- ✅ **Vec<T> CPU vector abstraction**
  - Supports AVX2/AVX512/NEON fallback
- ✅ **Thread-safe and decoder-compatible**

---

### 🧠 Architecture

```text
CPUAttention<T>::forward(...)
  └── cpu_paged_attention_forward<T>(...)
        ├── Vec<T> operations
        ├── rotary_embed(q, k)
        ├── qk_sim = q • k^T
        ├── softmax_lut()  (int8) / softmax_exp() (float)
        ├── Top-k / Top-p filter
        └── AV = Σ softmax(qk) * V
```

---

### 🧩 Integration with Decoder

This module is integrated by the decoder layer:

```cpp
CPUAttention<float> attention(config);
CPUAttentionInput<float> input = { ... };
attention.forward(input, output);
```

Input/output struct supports:

- float/int8 input Q/K/V
- KV tile cache
- optional rotary table
- LUT table
- output buffer for logits, attention weights

---

### ⚙️ INT8 Inference Stack

- Quantization via `int8_quant`
- GEMM via `dnnl_matmul_int8`
- Activation via LUT (GELU)
- Softmax via LUT + warp-style parallelism
- Top-k/p via fast filter kernel

---

### 📄 Example: Launch Kernel

```cpp
cpu_paged_attention_forward<float>(
  input.q, input.k, input.v,
  input.rotary_emb,
  input.kv_cache,
  input.lut,
  B, H, D, T,
  tile_size, top_k, top_p,
  input.temperature, input.causal, input.eos_token,
  output.out,
  output.attention_weights,
  output.logits
);
```

---

### 🔧 Performance Optimizations

- SIMD vectorization (Vec<T>) for matmul, qk, softmax
- Softmax via LUT (512-entry quantized exponential)
- per-tile loop unrolling and cache reuse
- Separate LUT logic from kernel to enable preloading

---

### 📂 Used by

- `int8_decoder.cpp`
- `CUDADecoder` fallback (for CPU mode)
- `cli/` → `generate_cli.py`, `chat_cli.py`, `generate_batch.py`
- `api/`, `web/` backend (auto CPU fallback via backend_router)

---

### 📘 References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [oneDNN INT8 Optimization Guide](https://github.com/oneapi-src/oneDNN)
- Vec<T> adapted from AVX2-style microkernel patterns

---

### ✅ Status

| Component                | Status |
|-------------------------|--------|
| Float32 Attention        | ✅ Stable |
| INT8 Attention           | ✅ Stable |
| Rotary / Causal / Masking| ✅ Ready |
| Top-k / Top-p Filter     | ✅ Integrated |
| EOS support              | ✅ Supported |
| Vectorization (Vec<T>)   | ✅ Cross-platform |
| Decoder integration      | ✅ Done |
