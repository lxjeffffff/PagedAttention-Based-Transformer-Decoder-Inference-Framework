# CLI: rerank model evaluation on JSONL path tree
# cli/rerank_eval.py
import argparse
from reranker.reranker import Reranker
from api.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--context", type=str, required=True, help="Rerank context prompt")
parser.add_argument("--beams", type=str, nargs='+', required=True, help="Beam candidates")
args = parser.parse_args()

reranker = Reranker("./reranker_model")
tokenizer = Tokenizer.get("gpt2")

scores = reranker.rerank_scores(args.context, args.beams)
best_idx = scores.index(max(scores))

print("\n=== Reranker Evaluation ===")
print("Context:", args.context)
print("Beams:")
for i, (beam, score) in enumerate(zip(args.beams, scores)):
    prefix = "-> " if i == best_idx else "   "
    print(f"{prefix}[{i}] {beam} (score: {score:.4f})")

print(f"\n✅ Best Beam: [{best_idx}] → {args.beams[best_idx]}")
