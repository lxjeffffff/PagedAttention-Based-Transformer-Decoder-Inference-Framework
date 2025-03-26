# cli/stream_cli.py
import time
from api.tokenizer import Tokenizer
from decoder.cuda_decoder import CUDADecoder

tokenizer = Tokenizer.get("gpt2")
decoder = CUDADecoder(12, 12, 64, 768, 50257, 2048)
decoder.load_weights("weights")

prompt = input("Prompt: ").strip()
input_ids = tokenizer.encode(prompt)

print("\nStreaming response:\n", end="", flush=True)
for _ in range(64):
    output_ids = []
    decoder.generate(input_ids, output_ids, max_gen_len=1)
    next_token = output_ids[-1]
    token_text = tokenizer.decode([next_token])
    print(token_text, end="", flush=True)
    input_ids.append(next_token)
    time.sleep(0.05)
print("\n\n✅ Done.")
