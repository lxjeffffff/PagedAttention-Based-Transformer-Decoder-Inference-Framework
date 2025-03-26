`web/` — Flask-Based Streaming Inference Server for LLM Decoding

This module provides a lightweight Flask web server for real-time and batch inference using a transformer decoder backend. It supports both CPU and GPU execution via backend routing, and includes token-by-token streaming via Server-Sent Events (SSE).

---

### 📦 Contents

| File              | Description |
|-------------------|-------------|
| `app.py`          | Flask app with route definitions (`/generate`, `/chat`, `/stream_chat_beam`, etc.) |
| `backend_router.py` | Backend selector for `CUDADecoder` and `INT8Decoder` |
| `config.py`       | Global config (e.g., `BACKEND_MODE = cpu/gpu/auto`) |
| `sse_utils.py`    | Utility to generate SSE-compatible HTTP responses |
| `README.md`       | This documentation |

---

### ⚙️ Supported Routes

| Method | Endpoint             | Description |
|--------|----------------------|-------------|
| `POST` | `/generate`          | One-shot generation for a prompt |
| `POST` | `/stream`            | Token-by-token streaming generation |
| `POST` | `/chat`              | Multi-turn chat stream with history |
| `POST` | `/stream_chat_beam`  | Beam search + reranker reranking stream |
| `POST` | `/generate_batch`    | Batch generation for multiple prompts |

---

### 🧠 Backend Routing (`backend_router.py`)

```python
decoder_gpu = CUDADecoder(...)
decoder_gpu.load_weights("weights")

decoder_cpu = INT8Decoder(...)
decoder_cpu.load_quantized_weights("weights/int8")

def select_backend():
    if BACKEND_MODE == "gpu": return decoder_gpu
    if BACKEND_MODE == "cpu": return decoder_cpu
    return decoder_cpu
```

- Set mode in `web/config.py`:

```python
BACKEND_MODE = "auto"  # or "gpu", "cpu"
```

---

### 📄 API Example: `/generate`

```json
POST /generate
{
  "prompt": "Explain AI",
  "max_tokens": 32,
  "temperature": 1.0
}
```

Response:

```json
{
  "input": "Explain AI",
  "output_ids": [15496, 703, ...],
  "output_text": "AI is a branch of computer science..."
}
```

---

### 📄 Streaming Example: `/stream_chat_beam`

```json
POST /stream_chat_beam
{
  "messages": ["Hello", "Tell me a story."],
  "beam_width": 4,
  "max_tokens": 32,
  "use_rerank": true
}
```

- Returns token-by-token SSE stream
- Selects best candidate via reranker
- Sends final sequence as stream

---

### 📂 Tokenizer & Reranker Integration

- `Tokenizer` is loaded via HuggingFace and cached
- Beam reranking is handled via `Reranker.select_best(...)`
- SSE utility `stream_sse()` wraps Python generators into compliant streaming output

---

### ✅ Server Startup

```bash
python web/app.py
# Listening at: http://localhost:8080
```

---

### ✅ System Design

```text
        Client Request
              ↓
        ┌────────────┐
        │  Flask API │
        └────┬───────┘
             ↓
       [select_backend()] ──→ CUDADecoder / INT8Decoder
             ↓
        decoder.generate()
             ↓
       Tokenizer.decode() → Response (stream or JSON)
```

---

### ✅ Status

| Component           | Status |
|---------------------|--------|
| REST endpoints       | ✅ Stable |
| SSE streaming        | ✅ Integrated |
| Beam reranking       | ✅ Supported |
| Tokenizer cache      | ✅ Active |
| Multi-backend support| ✅ via `backend_router.py` |

---

### 🛠️ Future Improvements

- Add OpenAPI spec (Swagger) for `/generate`, `/stream`
- Integrate `/rerank_scores` debug route
- Add `/tokenize`, `/detokenize` for frontend debugging
- Include frontend HTML demo (via `/static`)
