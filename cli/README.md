# `cli/` — Command-Line Toolkit for Transformer Inference & Evaluation

This directory provides a suite of command-line tools for testing, benchmarking, logging, and evaluating transformer model inference. Tools cover single and multi-turn generation, streaming outputs, beam reranking, batch decoding, and reranker training data generation.

---

### 📦 Contents

| File                    | Description |
|-------------------------|-------------|
| `generate_cli.py`       | Single-shot prompt generation |
| `chat_cli.py`           | Multi-turn interactive chat interface |
| `stream_cli.py`         | Simulates token-by-token streaming response |
| `log_cli.py`            | Chat interface with JSONL session logging |
| `generate_batch.py`     | Batch generation for multiple prompts from file |
| `rerank_eval.py`        | Score beam outputs using reranker and mark the best |
| `rerank_jsonl_builder.py` | Create JSONL training data for reranker |
| `cli_app.py`            | Unified CLI launcher for all tools |

---

### 🧠 Tool Overview

#### 🟢 `generate_cli.py`

```bash
python cli/generate_cli.py --prompt "What is a Transformer?" --max_tokens 64
```

- Generates output for a single prompt
- Supports temperature control

#### 🟢 `chat_cli.py`

```bash
python cli/chat_cli.py
```

- Multi-turn chat session
- Maintains message history
- Uses decoder + tokenizer

#### 🟢 `stream_cli.py`

```bash
python cli/stream_cli.py
```

- Stream tokens one at a time
- Simulates real-time output

#### 🟢 `log_cli.py`

```bash
python cli/log_cli.py
```

- Same as chat CLI but logs every round into a `.jsonl` file
- Useful for training data construction

#### 🟢 `generate_batch.py`

```bash
python cli/generate_batch.py --input prompts.txt --output outputs.jsonl
```

- Input: plain text file, one prompt per line
- Output: JSONL file with prompt + generated output

#### 🟢 `rerank_eval.py`

```bash
python cli/rerank_eval.py --context "Tell a joke" --beams "A" "B" "C"
```

- Scores all beam candidates
- Highlights the highest scoring candidate

#### 🟢 `rerank_jsonl_builder.py`

```bash
python cli/rerank_jsonl_builder.py
```

- Creates labeled reranker training data from beam outputs
- Outputs JSONL format with `label: 1` for gold path

#### 🟢 `cli_app.py`

```bash
python cli/cli_app.py generate --prompt "Hello"
```

- Master entry point to run all tools
- Automatically forwards arguments to the correct tool

---

### 🧩 Components Used

| Module         | Description |
|----------------|-------------|
| `Tokenizer`    | Used to encode/decode all input/output strings |
| `CUDADecoder`  | Main inference backend (weights must be loaded) |
| `INT8Decoder`  | Can be integrated with quantized CLI tools |
| `Reranker`     | Evaluates beam candidates, used in rerank tools |

---

### 📁 Directory Conventions

- All weights must be located under `weights/`
- Tokenizer is loaded via `"gpt2"` by default
- All decoder instances load weights on startup: `decoder.load_weights("weights")`

---

### ✅ Status

| Tool                   | Status |
|------------------------|--------|
| Prompt generation      | ✅ Stable |
| Streaming response     | ✅ Stable |
| Chat with context      | ✅ Stable |
| Reranker integration   | ✅ Ready |
| JSONL builder          | ✅ Integrated |
| CLI master launcher    | ✅ Supported |

---

### 📘 Example Workflow

```bash
# 1. Generate response
python cli/generate_cli.py --prompt "Explain transformers"

# 2. Run a chat session
python cli/chat_cli.py

# 3. Rerank beam outputs
python cli/rerank_eval.py --context "Explain AI" --beams "A" "B" "C"

# 4. Create training data
python cli/rerank_jsonl_builder.py

# 5. Use master CLI
python cli/cli_app.py chat
```
