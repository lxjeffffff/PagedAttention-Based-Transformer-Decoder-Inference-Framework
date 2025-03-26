`decoder/` — Modular Transformer Decoder for CUDA and INT8 Inference

This module implements a modular transformer decoder stack that supports both GPU-accelerated inference via CUDA and INT8 quantized inference via CPU (oneDNN). It includes components for token embedding, layer normalization, multi-layer MLP, tiled key-value cache, and attention execution (GPU or CPU backend).

---

### 📦 Contents

| File                      | Description |
|---------------------------|-------------|
| `cuda_decoder.hpp/cpp`    | `CUDADecoder<T>`: float-based decoder using GPU FlashAttention |
| `int8_decoder.hpp/cpp`    | `INT8Decoder`: int8 quantized decoder using CPU attention (oneDNN) |
| `decoder_block.hpp`       | A single decoder block: LayerNorm + Attention + MLP |
| `token_embedding.hpp`     | Token embedding layer with lookup and prefetch support |
| `layer_norm.hpp`          | Layer normalization implementation with weight loading |
| `mlp.hpp`                 | MLP module with 2-layer projection and ReLU activation |
| `README.md`               | This documentation |

---

### 🧠 Architecture

```text
TokenEmbedding → N × DecoderBlock → logits → sampling

DecoderBlock:
  ├── LayerNorm (pre-attention)
  ├── Attention (uses CUDA or CPU)
  ├── LayerNorm (post-attention)
  └── MLP (2-layer with activation)
```

---

### 🔄 Decoder Implementations

| Class           | Backend  | Precision | Notes |
|-----------------|----------|-----------|-------|
| `CUDADecoder<T>` | GPU (CUDA) | float32   | Uses FlashAttention + tile-based KV cache |
| `INT8Decoder`    | CPU (oneDNN) | int8     | Uses quantized linear + LUT softmax |

---

### 🚀 Key Features

- ✅ Modular transformer decoder for both GPU and CPU
- ✅ KV tile cache integrated (`KVTileCache<T>`, `KVTileCacheCPU<T>`)
- ✅ `DecoderBlock<T>` compatible with float/int8
- ✅ TokenEmbedding supports fast lookup and batching
- ✅ INT8 quantization tools included (`quantize_weights(...)`)
- ✅ Supports streaming and batch generation (`generate()` API)
- ✅ Easily extendable to multi-GPU / multi-thread setup

---

### 🧩 Integration with CLI / API / Web

| Component         | Role |
|------------------|------|
| `generate_cli.py` | CLI batch generation |
| `chat_cli.py`     | Multi-turn chat CLI |
| `api/router.py`   | REST + streaming APIs |
| `web/app.py`      | Flask server for web inference |
| `kv_cache/`       | External tile KV cache backend |
| `attention/`      | FlashAttention kernel invoked via `DecoderBlock` |

---

### 📄 Usage Example (CUDADecoder)

```cpp
CUDADecoder<float> decoder(...);
decoder.load_weights("weights/");
std::vector<int> input_ids = {...};
std::vector<int> output_ids;
decoder.generate(input_ids, output_ids, 64, 1.0f);
```

---

### 📄 Usage Example (INT8Decoder)

```cpp
INT8Decoder decoder(...);
decoder.load_quantized_weights("weights/int8");
decoder.generate(input_ids, output_ids, 64, 1.0f);
```

---

### 🧰 Quantization Pipeline

- FP32 → INT8 using `quantize_weights(...)`
- Embedding + MLP + LN weights all quantized layer-wise
- Weight files written as raw `.bin` (little endian)

---

### 🛠️ Weight Files Format

| File Path                     | Description |
|-------------------------------|-------------|
| `weights/embedding.bin`       | float or int8 token embeddings |
| `weights/layer_0/ln1.bin`     | LayerNorm gamma + beta |
| `weights/layer_0/mlp_fc1.bin` | MLP weights: FC1 |
| `weights/layer_0/mlp_fc2.bin` | MLP weights: FC2 |
| ...                           | Repeat per layer |

---

### ✅ Status

| Component       | Status |
|------------------|--------|
| Float32 decoder  | ✅ Stable |
| INT8 decoder      | ✅ Stable |
| MLP / LN modules  | ✅ Integrated |
| Weight loader     | ✅ Unified |
| KV cache support  | ✅ Cross-compatible |

---

### 📘 References

- FlashAttention: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- vLLM / HuggingFace transformer structure
- oneDNN: INT8 matmul acceleration
