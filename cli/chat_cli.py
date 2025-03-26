# CLI: conversation → response, stream / beam + rerank
# cli/chat_cli.py
from api.tokenizer import Tokenizer
from decoder.cuda_decoder import CUDADecoder

tokenizer = Tokenizer.get("gpt2")
decoder = CUDADecoder(num_layers=12, num_heads=12, head_dim=64, hidden_dim=768, vocab_size=50257, max_seq_len=2048)
decoder.load_weights("weights")

messages = []

print("Multi-turn Chat (type 'exit' to quit)\n")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() == "exit":
        break

    messages.append(f"User: {user_input}")
    context_text = " ".join(messages)
    input_ids = tokenizer.encode(context_text)

    output_ids = []
    decoder.generate(input_ids, output_ids, max_gen_len=64, temperature=1.0)
    reply = tokenizer.decode(output_ids)
    print("Assistant:", reply)
    messages.append(f"Assistant: {reply}")
