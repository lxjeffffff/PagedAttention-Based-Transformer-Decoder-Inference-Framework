# `weights/` — Transformer Model Weights Directory

This directory holds all pre-trained or quantized weights for the transformer decoder model, including token embeddings and per-layer components. These weights are loaded by `CUDADecoder` and `INT8Decoder` during runtime initialization.

> 📝 Note: This folder is currently empty — it will be populated with `.bin` files or `.npz` files as required.

---

### 📦 Structure Overview

| Path                              | Description                                 |
|-----------------------------------|---------------------------------------------|
| `embedding.bin`                   | Token embedding matrix (float or int8)      |
| `layer_0/` ~ `layer_N/`           | Decoder block weights (per layer)           |
| `int8/`                           | Quantized version of weights for INT8Decoder |
| `float32/`                        | Optional full-precision model subfolder     |

---

### 📂 Per-Layer Files (within `layer_N/`)

| File Name        | Content                             | Shape (example)             |
|------------------|--------------------------------------|-----------------------------|
| `ln1.bin`         | LayerNorm1 gamma + beta             | `[hidden_dim * 2]`          |
| `attn_wq.bin`     | Attention Wq matrix (optional)       | `[hidden_dim, head_dim]`    |
| `attn_wk.bin`     | Attention Wk matrix                  | same                        |
| `attn_wv.bin`     | Attention Wv matrix                  | same                        |
| `attn_wo.bin`     | Attention output projection (Wo)     | `[head_dim, hidden_dim]`    |
| `ln2.bin`         | LayerNorm2 gamma + beta             | `[hidden_dim * 2]`          |
| `mlp_fc1.bin`     | MLP feedforward layer 1             | `[hidden_dim, inter_dim]`   |
| `mlp_fc2.bin`     | MLP feedforward layer 2             | `[inter_dim, hidden_dim]`   |
| `mlp_biases.bin`  | Bias vector for fc1 and fc2         | `[inter_dim + hidden_dim]`  |

> You can omit unused files if your decoder uses fused projection layers.

---

### 🧠 Used By

| Decoder           | Loader Method                       | Notes                         |
|------------------|-------------------------------------|-------------------------------|
| `CUDADecoder`     | `load_weights("weights/")`          | Loads `embedding.bin` + all `layer_N/` |
| `INT8Decoder`     | `load_quantized_weights("weights/int8/")` | INT8 weights must be pre-quantized |
| `quantize_weights()` | Writes INT8 weights to `weights/int8/` | See `INT8Decoder::quantize_weights(...)` |

---

### 🔄 Weight Format

- All files are raw `.bin` files (binary little-endian)
- Each file is expected to be a flat buffer of `float32` or `int8_t`
- Total size = `num_elements * sizeof(type)`
- File names must exactly match loader expectations

---

### 🛠️ Sample Workflow

```bash
# Example structure
weights/
├── embedding.bin
├── layer_0/
│   ├── ln1.bin
│   ├── mlp_fc1.bin
│   └── ...
├── layer_1/
│   └── ...
├── int8/
│   └── embedding.bin
```

Then, in Python/CLI/API:

```python
decoder.load_weights("weights/")
```

---

### ✅ Status

| Component         | Status |
|------------------|--------|
| Directory created | ✅ |
| Structure defined | ✅ |
| Files present     | ⚠️ Empty for now |
| Ready for loading | ✅ |

---

### 📘 References

- Weight format consistent with `decoder_block::load_weights()`
- Compatible with `load_vector_from_file()` logic in C++
