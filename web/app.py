# Flask app
# web/app.py
from flask import Flask, request, jsonify
from web.backend_router import select_backend, get_tokenizer
from web.sse_utils import stream_sse
from reranker.reranker import Reranker

app = Flask(__name__)
tokenizer = get_tokenizer()
reranker = Reranker("./reranker_model")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = int(data.get("max_tokens", 64))
    temperature = float(data.get("temperature", 1.0))

    input_ids = tokenizer.encode(prompt)
    decoder = select_backend()

    output_ids = []
    decoder.generate(input_ids, output_ids, max_tokens, temperature)
    output_text = tokenizer.decode(output_ids)

    return jsonify({
        "input": prompt,
        "output_ids": output_ids,
        "output_text": output_text
    })

@app.route("/stream", methods=["POST"])
def stream():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = int(data.get("max_tokens", 64))
    temperature = float(data.get("temperature", 1.0))

    input_ids = tokenizer.encode(prompt)
    decoder = select_backend()

    def generate_tokens():
        context = list(input_ids)
        for _ in range(max_tokens):
            output_ids = []
            decoder.generate(context, output_ids, 1, temperature)
            token_id = output_ids[-1]
            context.append(token_id)
            yield {
                "token": token_id,
                "text": tokenizer.decode([token_id])
            }

    return stream_sse(generate_tokens)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    context_text = " ".join(messages)
    max_tokens = int(data.get("max_tokens", 64))
    temperature = float(data.get("temperature", 1.0))

    input_ids = tokenizer.encode(context_text)
    decoder = select_backend()

    def stream_reply():
        context = list(input_ids)
        for _ in range(max_tokens):
            output_ids = []
            decoder.generate(context, output_ids, 1, temperature)
            token_id = output_ids[-1]
            context.append(token_id)
            yield {
                "token": token_id,
                "text": tokenizer.decode([token_id])
            }

    return stream_sse(stream_reply)

@app.route("/stream_chat_beam", methods=["POST"])
def stream_chat_beam():
    data = request.json
    messages = data.get("messages", [])
    beam_width = int(data.get("beam_width", 4))
    max_tokens = int(data.get("max_tokens", 64))
    temperature = float(data.get("temperature", 1.0))
    use_rerank = bool(data.get("use_rerank", True))

    context_text = " ".join(messages)
    prompt_ids = tokenizer.encode(context_text)
    decoder = select_backend()

    beams = []
    for _ in range(beam_width):
        output_ids = []
        decoder.generate(prompt_ids, output_ids, max_tokens, temperature)
        beams.append(tokenizer.decode(output_ids))

    best_idx = reranker.select_best(context_text, beams) if use_rerank else 0
    best_tokens = tokenizer.encode(beams[best_idx])

    def stream_beam_tokens():
        for tok in best_tokens:
            yield {
                "token": tok,
                "text": tokenizer.decode([tok])
            }

    return stream_sse(stream_beam_tokens)

@app.route("/generate_batch", methods=["POST"])
def generate_batch():
    data = request.json
    prompts = data.get("prompts", [])
    max_tokens = int(data.get("max_tokens", 64))
    temperature = float(data.get("temperature", 1.0))

    decoder = select_backend()
    outputs = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        output_ids = []
        decoder.generate(input_ids, output_ids, max_tokens, temperature)
        output_text = tokenizer.decode(output_ids)
        outputs.append({
            "prompt": prompt,
            "output_ids": output_ids,
            "output_text": output_text
        })

    return jsonify({"results": outputs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
