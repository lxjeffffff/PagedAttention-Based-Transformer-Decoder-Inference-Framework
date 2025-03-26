PagedAttention-Based Transformer Decoder Inference Framework

This project is a modular, production-ready framework for running high-performance transformer decoder inference powered by **PagedAttention**.

It supports both GPU (FlashAttention-style fused kernels) and CPU (INT8 vectorized tile kernels with oneDNN). The system integrates token streaming, beam search, reranking, KV tile caching, and REST/CLI/Web-based access.

---

## 🚀 Core Highlights

- ✅ **PagedAttention** with tile-based KV caching
- ✅ GPU FlashAttention-style fused kernels (`paged_flash_attention_kernel_fused`)
- ✅ CPU INT8 inference with AVX2 + LUT softmax + oneDNN
- ✅ Supports single/multi-turn generation, streaming, rerank
- ✅ Unified `CUDADecoder` and `INT8Decoder` APIs
- ✅ HuggingFace tokenizer + caching
- ✅ Beam search reranker + JSONL export + finetuning

---

## 📁 Project Structure Overview

```
project_root/
├── api/           # FastAPI-based inference server
├── web/           # Flask SSE web interface
├── cli/           # CLI tools for generate, chat, batch, rerank
├── decoder/       # Core decoder blocks (CUDADecoder, INT8Decoder)
├── attention/     # GPU FlashAttention kernels (PagedAttention)
├── attention_cpu/ # CPU kernels (INT8 + vectorized softmax)
├── kv_cache/      # KV tile cache with page table (GPU + CPU)
├── reranker/      # Beam reranker + finetuning tools
├── config/        # Model architecture + runtime configuration
├── weights/       # Placeholder for model weights
├── include/       # pybind11 headers
├── src/           # pybind11 binding logic
├── requirements.txt
├── CMakeLists.txt
├── setup.py
└── README.md      # You are here
```

---

## 🧠 What is PagedAttention?

PagedAttention refers to an attention computation mechanism where the KV cache is broken into **tiles (pages)**, and only necessary tiles are fetched during streaming inference. This:

- Reduces memory reads
- Enables longer context windows
- Supports efficient beam search and reranker interaction
- Powers **streaming + high throughput decoding**

This project implements PagedAttention for both GPU and CPU backends, including:

- Warp-level softmax and AV cache reuse (GPU)
- Tile-wise prefetch and masking (CPU INT8)
- Beam-aware prefetching
- Tile eviction + LRU cache on CPU

---

## 📦 Installation

### ✅ Python dependencies

```bash
pip install -r requirements.txt
```

Includes:
- `transformers`, `fastapi`, `pybind11`, `uvicorn`, `flask`, `datasets`, `matplotlib` etc.

---

### ⚙️ C++/CUDA build (CMake or setup.py)

#### Option 1: CMake build

```bash
mkdir build && cd build
cmake ..
make -j
```

#### Option 2: Python install

```bash
pip install .
```

Builds and installs `llm_decoder` Python module via pybind11 + C++/CUDA

---

## 🧪 CLI Usage

```bash
python cli/generate_cli.py --prompt "Hello"
python cli/chat_cli.py
python cli/generate_batch.py
python cli/rerank_eval.py
```

Unified launcher:

```bash
python cli/cli_app.py stream
```

---

## 🌐 Web / API Inference

### FastAPI server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Flask SSE server

```bash
python web/app.py
```

Supports:

- `/generate`
- `/stream`
- `/stream_chat_beam`
- `/generate_batch`

---

## 📄 Configuration Files

Located in `config/`:

| File                     | Purpose |
|--------------------------|---------|
| `model_config.yaml`      | Model layers, heads, dim, tokenizer |
| `runtime_config.yaml`    | Temperature, top_k, backend mode |
| `chat_template.json`     | Assistant prompt + multi-turn format |
| `weight_paths.yaml`      | Per-layer weight locations |
| `paths.json`             | Optional overrides |

---

## 💾 Weights Format

Expected under `weights/`:

```
weights/
├── embedding.bin
├── layer_0/
│   ├── ln1.bin
│   ├── mlp_fc1.bin
│   └── ...
├── int8/
│   └── embedding.bin (quantized)
```

Used by `decoder.load_weights(...)` or `int8_decoder.load_quantized_weights(...)`

---

## 🧩 Integration Points

- `api/` and `web/` → inference services
- `decoder/` → exposes `generate(...)` for float + int8
- `kv_cache/` → provides paged tile caching with LRU
- `attention/` → GPU FlashAttention (paged)
- `attention_cpu/` → CPU attention kernel (paged)
- `reranker/` → rerank beam candidates, tree visualization, JSONL training

---

## ✅ Status

| Component         | Status |
|-------------------|--------|
| PagedAttention (GPU) | ✅ |
| PagedAttention (CPU) | ✅ |
| CUDA inference    | ✅ Stable |
| INT8 CPU inference | ✅ Stable |
| CLI tools         | ✅ Integrated |
| API / Web         | ✅ Streaming ready |
| Reranker          | ✅ Supported |
| Pybind11 binding  | ✅ Working |
| Beam search       | ✅ with rerank |
| Weights           | ✅ (user-supplied .bin) |

---

## 📘 License & Acknowledgements

- FlashAttention concepts inspired by [vLLM](https://github.com/vllm-project/vllm)
- Beam reranking via Transformers + BERT-based scoring
- Quantization powered by oneDNN (INT8 matmul + LUT softmax)
- Tokenizer from HuggingFace (`AutoTokenizer`)
