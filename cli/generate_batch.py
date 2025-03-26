# cli/generate_batch.py
import argparse
from api.tokenizer import Tokenizer
from decoder.cuda_decoder import CUDADecoder

tokenizer = Tokenizer.get("gpt2")
decoder = CUDADecoder(12, 12, 64, 768, 50257, 2048)
decoder.load_weights("weights")

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Text file with one prompt per line")
parser.add_argument("--output", type=str, default="output.jsonl")
parser.add_argument("--max_tokens", type=int, default=64)
args = parser.parse_args()

with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
    for line in fin:
        prompt = line.strip()
        if not prompt:
            continue
        input_ids = tokenizer.encode(prompt)
        output_ids = []
        decoder.generate(input_ids, output_ids, args.max_tokens)
        output_text = tokenizer.decode(output_ids)
        fout.write(json.dumps({"prompt": prompt, "output": output_text}, ensure_ascii=False) + "\n")

print(f"Batch generation complete → {args.output}")
