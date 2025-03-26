# `reranker/` — Path Scoring & Reranking Module for Beam Search

This module implements a candidate reranker for transformer decoding. It evaluates multiple generated outputs (typically from beam search or sampling) and selects the best candidate based on learned or heuristic scoring.

The module also includes support for reranker training (`finetune.py`), data export (`export_jsonl.py`), and tree-based visualization of candidate paths (`plot_tree.py`).

---

### 🖋️ Author
Jeffrey
🌐 GitHub: [lxjeffffff](https://github.com/lxjeffffff)

---

### 📦 Contents

| File                   | Description |
|------------------------|-------------|
| `reranker.hpp/cpp`     | `Reranker` class with `rerank_scores()` and `select_best()` |
| `reranker_model.hpp/cpp` | Simulated or learned reranker model (`score(context, candidate)`) |
| `export_jsonl.py`      | Export labeled candidate pairs as JSONL for training |
| `finetune.py`          | Fine-tune a transformer-based reranker (BERT) using HuggingFace |
| `plot_tree.py`         | Visualize beam paths as a tree structure |
| `README.md`            | This documentation |

---

### 🧠 How It Works

```text
[Decoder] → [Beam outputs] → [Reranker]
                               ↓
         ┌─────────────── rerank_scores() ───────────────┐
         ↓                                                ↓
[Candidate1] [Candidate2] ... [CandidateN]        → select_best()
                                                         ↓
                                                 Final output
```

- The reranker receives a `context` prompt and a list of `candidate` completions.
- It returns a list of float scores (`rerank_scores()`), or the index of the best candidate (`select_best()`).

---

### 🚀 Features

- ✅ Score-based reranking from multiple outputs
- ✅ Trivial model implementation with deterministic hash + randomness
- ✅ Pluggable model backend (`RerankerModel`)
- ✅ HuggingFace-based training with `finetune.py`
- ✅ JSONL export for beam candidates with labels
- ✅ Candidate path tree visualization

---

### 🧩 Class Overview

#### `Reranker`

```cpp
class Reranker {
public:
    Reranker(const std::string& model_path);
    std::vector<float> rerank_scores(const std::string& context, const std::vector<std::string>& candidates);
    int select_best(const std::string& context, const std::vector<std::string>& candidates);
};
```

#### `RerankerModel`

```cpp
class RerankerModel {
public:
    float score(const std::string& context, const std::string& candidate);
};
```

Current implementation is hash-based scoring, but can be swapped with a trained BERT reranker.

---

### 📄 Training

Use `finetune.py` to fine-tune a `BertForSequenceClassification` reranker using HuggingFace Transformers.

```bash
python reranker/finetune.py
```

This loads `train.jsonl` (exported from `export_jsonl.py`) in the format:

```json
{"context": "Explain LLMs", "candidate": "A large language model is...", "label": 1}
```

---

### 📄 Export Training Data

```bash
python reranker/export_jsonl.py
```

This creates a labeled JSONL dataset for reranker fine-tuning from beam outputs.

---

### 📄 Tree Visualization

```bash
python reranker/plot_tree.py
```

Draws a visual tree of candidate beams using `networkx` and `matplotlib`. Input is a JSON file like:

```json
{
  "context": ["beam1", "beam2"],
  "beam1": ["beam1_1", "beam1_2"],
  "beam2": []
}
```

Output: `rerank_tree.png`

---

### ✅ API Integration Example

Used in `stream_chat_beam` (Flask or FastAPI):

```python
idx = reranker.select_best(prompt, beams)
best = beams[idx]
```

---

### ✅ CLI Integration

Used in `cli/rerank_eval.py`:

```bash
python cli/rerank_eval.py --context "Tell a joke" --beams "A", "B", "C"
```

---

### 📘 References

- BERT-based reranking: [ColBERT, monoBERT, etc.]
- Beam Search in transformers
- Reranking in information retrieval (IR)

---

### ✅ Status

| Component      | Status |
|----------------|--------|
| Reranker class | ✅ Stable |
| Training logic | ✅ Ready |
| JSONL export   | ✅ Verified |
| Tree plot      | ✅ Functional |
| Hash-based score | ✅ Simulated (default) |
| Pluggable model | ✅ Replaceable with HuggingFace BERT |
