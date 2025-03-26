# CLI: prompt → token output, support --mode
# cli/generate_cli.py
import argparse
from api.tokenizer import Tokenizer
from decoder.cuda_decoder import CUDADecoder

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate from")
parser.add_argument("--max_tokens", type=int, default=64)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

tokenizer = Tokenizer.get("gpt2")
decoder = CUDADecoder(num_layers=12, num_heads=12, head_dim=64, hidden_dim=768, vocab_size=50257, max_seq_len=2048)
decoder.load_weights("weights")

input_ids = tokenizer.encode(args.prompt)
output_ids = []
decoder.generate(input_ids, output_ids, args.max_tokens, args.temperature)
output_text = tokenizer.decode(output_ids)

print("=== Prompt ===")
print(args.prompt)
print("\n=== Completion ===")
print(output_text)
