# `api/` — REST & Streaming Inference API for Transformer Decoding

This module provides a FastAPI-based RESTful and streaming interface for transformer-based language model inference. It supports single-turn and multi-turn generation, beam search reranking, and integrates with tokenizer, decoder, and reranker modules.

---

### 🖋️ Author
Jeffrey
🌐 GitHub: [lxjeffffff](https://github.com/lxjeffffff)

---

### 📦 Contents

| File              | Description |
|-------------------|-------------|
| `main.py`         | FastAPI app entry point with route inclusion |
| `router.py`       | Core inference routes (`/generate`, `/stream`, `/chat`) |
| `schema.py`       | Pydantic request/response schemas |
| `tokenizer.py`    | Tokenizer wrapper using HuggingFace, with caching and multi-language support |
| `README.md`       | This documentation |

---

### 🚀 Features

- ✅ FastAPI-based JSON + streaming (`text/event-stream`) endpoints
- ✅ Supports single-prompt and multi-turn conversational input
- ✅ Integrated `CUDADecoder` for GPU inference
- ✅ Optional reranker selection with beam search
- ✅ HuggingFace tokenizer with encode/decode cache
- ✅ Can be deployed as REST API or streaming chat service

---

### 📄 API Endpoints

| Method | Endpoint              | Description                        |
|--------|------------------------|------------------------------------|
| `POST` | `/generate`            | One-shot generation (text or token input) |
| `POST` | `/stream_generate`     | Streaming generation (token-by-token)     |
| `POST` | `/stream_chat`         | Multi-turn chat generation stream         |
| `POST` | `/stream_chat_beam`    | Beam search with reranker + streaming     |

---

### 📚 Example Request: `/generate`

```json
POST /generate
{
  "prompt": "What is a transformer model?",
  "max_tokens": 32,
  "temperature": 0.9
}
```

**Response:**

```json
{
  "output_ids": [15496, 703, ...],
  "output_text": "A transformer is a type of deep learning model..."
}
```

---

### 🧠 Tokenizer Wrapper (`tokenizer.py`)

```python
tokenizer = Tokenizer.get("gpt2")
ids = tokenizer.encode("Hello world!")
text = tokenizer.decode(ids)
```

- Caches encode/decode results using `@lru_cache`
- Thread-safe, multi-language support (`get("gpt2"), get("xlm-roberta-base")`)
- Compatible with CLI, Web, and streaming APIs

---

### 🧩 Schema Definition (`schema.py`)

```python
class GenerateRequest(BaseModel):
    prompt: Optional[str]
    input_ids: Optional[List[int]]
    max_tokens: int = 32
    temperature: float = 1.0
```

All requests are validated with Pydantic before reaching your decoder.

---

### 🔄 Internals

| Component  | Role                              |
|------------|-----------------------------------|
| `CUDADecoder` | Inference engine (loaded at startup) |
| `Reranker` | Beam rerank scoring (`select_best()`)   |
| `Tokenizer` | HuggingFace tokenizer (cached)     |

---

### 🧪 Streaming Example (`/stream_chat`)

```bash
curl -X POST http://localhost:8000/stream_chat \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Hello", "What is AI?"], "max_tokens": 64}'
```

**SSE Response:**
```json
data: {"token": 15496, "text": "AI"}
data: {"token": 703, "text": " is"}
...
data: {"token": null, "finish_reason": "eos"}
```

---

### ✅ Status

| Component             | Status |
|------------------------|--------|
| REST generation        | ✅ Stable |
| Streaming (SSE)        | ✅ Stable |
| Beam rerank support    | ✅ Enabled |
| Tokenizer integration  | ✅ HuggingFace, cached |
| API schema validation  | ✅ Pydantic |
| Production deployment  | ✅ Ready with `uvicorn main:app` |

---

### 📘 Usage

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

### 🛠️ To Do / Ideas

- Add `/rerank_scores` debug route
- Add `/tokenize` and `/detokenize` endpoints
- Integrate `runtime_config.yaml` for dynamic API control
- Add rate limiting & request logging
- Integrated `INT8Decoder` for CPU inference
