`config/` — Centralized Configuration for Model, Runtime, and Paths

This directory contains all configuration files required to initialize, run, and manage the transformer-based LLM system across CLI, API, and Web applications. It includes model architecture, runtime behavior, weight paths, tokenizer info, and prompt templates for chat-based generation.

---

### 📦 Contents

| File                    | Description |
|-------------------------|-------------|
| `model_config.yaml`     | Defines the architecture of the transformer model |
| `runtime_config.yaml`   | Controls runtime settings: generation, backend, sampling |
| `weight_paths.yaml`     | Maps weight files and KV cache configuration |
| `paths.json`            | Optional short-form path alias (e.g. tokenizer, weights) |
| `chat_template.json`    | Defines chat prompt format, role names, and default assistant message |
| `README.md`             | This documentation |

---

### 🔧 `TODO`

- Fields need to be injected into the decoder, API, and several modules
