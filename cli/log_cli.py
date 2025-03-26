# cli/log_cli.py
import json
import datetime
from api.tokenizer import Tokenizer
from decoder.cuda_decoder import CUDADecoder

tokenizer = Tokenizer.get("gpt2")
decoder = CUDADecoder(12, 12, 64, 768, 50257, 2048)
decoder.load_weights("weights")

session = []
logfile = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

print("Chat CLI with Logging (type 'exit' to quit)")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() == "exit":
        break

    session.append({"role": "user", "content": user_input})
    prompt = " ".join([msg["content"] for msg in session])
    input_ids = tokenizer.encode(prompt)
    output_ids = []
    decoder.generate(input_ids, output_ids, max_gen_len=64)
    reply = tokenizer.decode(output_ids)
    print("Assistant:", reply)
    session.append({"role": "assistant", "content": reply})

    with open(logfile, "a", encoding="utf-8") as f:
        f.write(json.dumps({"messages": session[-2:]}) + "\n")
