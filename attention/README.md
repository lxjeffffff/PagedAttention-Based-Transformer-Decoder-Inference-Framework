# `attention/` — High-Performance FlashAttention Kernels for Transformer Decoders

This module implements a high-performance FlashAttention-like tile-based attention kernel optimized for modern GPU inference. It supports both **prefill** and **decode** modes, **tile-level caching**, **streaming**, and **beam search** scenarios.

---

### 📦 Contents

| File | Description |
|------|-------------|
| `attention_config.hpp` | Configuration structure for kernel launch (head count, tile size, etc.) |
| `attention_cuda.hpp/cpp` | Main `AttentionCUDA` class that dispatches kernels based on precision and mode |
| `attention_tile_launcher.hpp` | Dispatch wrapper for launching paged fused kernels |
| `attention_kernel_utils.cuh` | Utility functions (e.g. rotary embedding, masking) for use in GPU kernels |
| `paged_flash_attention_kernel_fused.cu` | Fused attention kernel (prefill, batch QK^T + AV) |
| `paged_flash_attention_kernel_fused_overlap.cu` | Overlapped streaming kernel (for decode mode) |
| `top_k_top_p_filter.cuh` | Top-k / Top-p sampling filtering kernel logic |
| `warp_softmax.cuh` | Warp-level numerically stable softmax implementation |
| `vec.hpp` | Template `Vec<T>` abstraction for GPU vector math with float, half, int8 types |

---

### 🚀 Features

- ✅ **Fused QK^T + AV computation** in single kernel
- ✅ **Rotary positional embedding** (RoPE) integration
- ✅ **KV tile cache prefetch and reuse**
- ✅ **Beam search support with reranking scores**
- ✅ **Streaming decode mode with tile overlap**
- ✅ **Top-k / Top-p filtering**
- ✅ **Multi-precision support**: `float32`, `__half`, `__nv_bfloat16`, `int8_t`

---

### ⚙️ Runtime Modes

| Mode      | Description                     |
|-----------|---------------------------------|
| Prefill   | Batch prefill mode (Q vs. Kcache) for initial prompt |
| Decode    | Streaming generation (1-step Q vs. K cache tile) |
| Overlap   | Enables tile pipelining between QK^T and AV steps |

The correct kernel is selected automatically by `AttentionCUDA::forward(...)`.

---

### 🧠 Integration with Decoder

This module is invoked by `DecoderBlock` inside `CUDADecoder`. Q projection is done upstream, and `AttentionCUDA` operates on pre-projected queries and cached K/V:

```cpp
attention_cuda_.forward(
    q, out,
    B, H, D, T,
    beam_ids,
    kv_cache,
    rotary_emb,
    is_prefill,
    use_fp16,
    use_overlap
);
```

---

### 🗂️ Kernel Launch Flow

```cpp
→ AttentionCUDA::forward(...)
   → dispatch to AttentionTileLauncher::launch<T>
      → launch paged_flash_attention_kernel_fused[_overlap]<T>
         → use KVTileCache + rotary + warp_softmax + sampling
```

---

### 📐 Dependencies

- CUDA ≥ 11.4
- Supports warp-level sync (`__shfl_sync`)
- Assumes KV cache is pre-allocated externally (`KVTileCache`)
- Compatible with cuBLAS, TRT, OpenCV-DNN downstream modules

---

### 📄 Example Kernel API

```cpp
__global__ void paged_flash_attention_kernel_fused<T>(
    const float* q,
    KVTileCache<T>* kv_cache,
    float* output,
    float* rerank_scores,
    int B, int T, int H, int D,
    const float* rotary_emb,
    const int* beam_ids,
    float temperature,
    int top_k,
    float top_p
);
```

---

### 📊 Performance Notes

- Fused kernel significantly reduces memory roundtrips vs. naïve QK/softmax/AV
- Vectorized `Vec<T>` abstraction auto-optimizes half/int8 path
- Beam tile fusion in `overlap` kernel supports streaming + reranking efficiently

---

### ✅ Status

| Component                 | Status |
|--------------------------|--------|
| AttentionCUDA kernel dispatch | ✅ Stable |
| Fused + overlap kernel         | ✅ Optimized |
| Top-k/top-p filtering          | ✅ Ready |
| Rotary embedding               | ✅ Ready |
| Warp-level softmax             | ✅ Verified |
| Reranker integration support   | ✅ Supported |

---

### 📂 Used By

- `decoder/DecoderBlock`
- `CUDADecoder`
- `api/router.py` (indirectly via decoder)
- `web/app.py` (via backend_router)

---

### 📘 References

- FlashAttention: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- vLLM: Paged KV Cache and attention routing
- NVIDIA warp-softmax examples
