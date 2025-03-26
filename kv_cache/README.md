# `kv_cache/` — Tile-based Key/Value Cache for Paged Attention

This module implements an efficient, tile-based KV cache system to support Paged Attention for both GPU and CPU backends. It includes page table mapping, tile registration, dynamic eviction (LRU), and host-device synchronization.

---

### 📦 Contents

| File                          | Description |
|-------------------------------|-------------|
| `kv_tile_cache.hpp/cpp`       | GPU-side tile cache manager using CUDA memory |
| `kv_tile_cache_cpu.hpp/cpp`   | CPU-side tile cache with LRU eviction and multithread-safe access |
| `page_table.hpp/cpp`          | Linearized 3D-to-1D page table implementation with CUDA sync |
| `README.md`                   | This documentation |

---

### 🚀 Features

- ✅ GPU tile-based KV cache with page table and `cudaMalloc` memory
- ✅ CPU tile cache with multi-threading (`shared_mutex`) and LRU eviction
- ✅ Beam-aware multi-sequence tile allocation
- ✅ PageTable structure for fast lookup and device sync
- ✅ Optional disk persistence (load/save cache state)
- ✅ Kernel-level tile `get()` and `get_write_ptr()` for CUDA

---

### 🧠 Tile Cache Architecture

```text
[DecoderBlock] → [AttentionCUDA]
                  ↓
            [KVTileCache]
         ┌────────┴─────────┐
         │    GPU CUDA      │
         │ KVTileCache<T>   │  ←→ PageTable
         └────────┬─────────┘
                  │
         ┌────────┴─────────┐
         │    CPU version   │
         │ KVTileCacheCPU<T>│ (AVX / LRU / thread-safe)
         └──────────────────┘
```

---

### 🗂️ GPU: `KVTileCache<T>`

```cpp
// CUDA-side buffer management
void init(int num_pages, int tile_size, int head_dim);
const T* get(int beam_id, int head_id, int tile_id, char type);
void register_tile(...);  // Allocates a tile if needed, uses LRU
```

- Internally manages `key_buffer_` and `value_buffer_` via `cudaMalloc`
- Uses `PageTable` to track [beam, head, tile] → page ID
- Supports eviction via LRU strategy (`evict_if_needed()`)

---

### 🧩 Page Table

- PageTable is a linear 3D-to-1D mapping
- `lookup(beam, head, tile)` returns device-side page ID
- `assign()`, `remove()`, `clear()` used for tile management
- `sync_to_gpu()` uploads entire host table for prefetch

---

### 🧩 CPU: `KVTileCacheCPU<T>`

```cpp
KVTileCacheCPU(int max_size, int tile_size);
void put(...);              // Insert/update tile
const T* get(...);          // Fetch tile (thread-safe)
void save(path);            // Optional: persist cache to file
```

- Backed by `unordered_map<TileIndex, std::vector<T>>`
- Supports batch put/get, LRU eviction
- Thread-safe via `shared_mutex` (multiple readers, one writer)
- Compatible with INT8/float and AVX-accelerated vector ops

---

### 📄 Example (CUDA-side tile access)

```cpp
__device__ const float* k_tile = kv_cache.get(beam_id, head_id, tile_id, 'k');
__device__ float* v_write_ptr = kv_cache.get_write_ptr(beam_id, head_id, tile_id, 'v');
```

---

### 📄 Example (CPU-side usage)

```cpp
KVTileCacheCPU<float> cpu_cache(1024, 128);
cpu_cache.put(batch, head, tile_id, raw_ptr);
const float* tile_data = cpu_cache.get(batch, head, tile_id);
```

---

### 📂 Used By

- `attention/` kernels: fused FlashAttention
- `attention_cpu/`: Paged CPU Attention with LUT softmax
- `decoder/`: DecoderBlock KV management
- `cli/`, `web/`: backend inference

---

### 🛠️ Dependencies

- CUDA 11.x+
- C++17 (`shared_mutex`, `unordered_map`)
- Compatible with Linux/Windows

---

### ✅ Status

| Component      | Status |
|----------------|--------|
| CUDA buffer mgmt  | ✅ |
| PageTable        | ✅ |
| CPU tile cache   | ✅ |
| LRU eviction     | ✅ |
| Thread-safe cache| ✅ |
| Save/load support| ✅ |

---

### 📘 References

- [vLLM: Paged KV Cache](https://github.com/vllm-project/vllm)
- [Rotary Embedding for Attention](https://arxiv.org/abs/2104.09864)
- [Efficient Beam Search Caching](https://arxiv.org/abs/2209.10655)
