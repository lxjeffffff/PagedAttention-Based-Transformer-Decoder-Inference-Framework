# cli/rerank_jsonl_builder.py
import json
from api.tokenizer import Tokenizer
from decoder.cuda_decoder import CUDADecoder

tokenizer = Tokenizer.get("gpt2")
decoder = CUDADecoder(12, 12, 64, 768, 50257, 2048)
decoder.load_weights("weights")

prompts = [
    "introduce Transformer",
    "what is Quantum Computing?",
    "Write a poem about spring"
]

jsonl_path = "rerank_dataset.jsonl"
beam_width = 4

with open(jsonl_path, "w", encoding="utf-8") as f:
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        candidates = []
        for _ in range(beam_width):
            output_ids = []
            decoder.generate(input_ids, output_ids, max_gen_len=64)
            decoded = tokenizer.decode(output_ids)
            candidates.append(decoded)

        # Mark the first one as a positive sample, and the others as negative samples (the strategy can be adjusted)
        for i, cand in enumerate(candidates):
            entry = {
                "context": prompt,
                "candidate": cand,
                "label": 1 if i == 0 else 0
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"JSONL saved: {jsonl_path}")
