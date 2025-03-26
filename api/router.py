from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.schema import GenerateRequest, GenerateResponse, StreamChatRequest, BeamStreamRequest
from decoder.cuda_decoder import CUDADecoder
from reranker.reranker import Reranker
from api.tokenizer import Tokenizer

import time
import json
from typing import Iterator

router = APIRouter()

decoder = CUDADecoder(num_layers=12, num_heads=12, head_dim=64, hidden_dim=768, vocab_size=50257, max_seq_len=2048)
decoder.load_weights("weights")
reranker = Reranker("./reranker_model")
tokenizer = Tokenizer("gpt2")

@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    input_ids = req.input_ids or tokenizer.encode(req.prompt)
    output_ids = []
    decoder.generate(input_ids, output_ids, req.max_tokens, req.temperature)
    output_text = tokenizer.decode(output_ids)
    return GenerateResponse(output_ids=output_ids, output_text=output_text)

@router.post("/stream_generate")
def stream_generate(req: GenerateRequest):
    def token_stream() -> Iterator[str]:
        context = req.input_ids or tokenizer.encode(req.prompt)
        for i in range(req.max_tokens):
            output_ids = []
            decoder.generate(context, output_ids, 1, req.temperature)
            next_token = output_ids[-1]
            yield json.dumps({"token": next_token, "text": tokenizer.decode([next_token])}) + "\n"
            context.append(next_token)
            time.sleep(0.05)
        # add EOS marker
        yield json.dumps({"token": None, "finish_reason": "eos"}) + "\n"
    return StreamingResponse(token_stream(), media_type="application/json")

@router.post("/stream_chat")
def stream_chat(req: StreamChatRequest):
    def chat_stream() -> Iterator[str]:
        context_text = " ".join(req.messages)
        tokens = tokenizer.encode(context_text)
        for _ in range(req.max_tokens):
            output_ids = []
            decoder.generate(tokens, output_ids, 1, req.temperature)
            next_token = output_ids[-1]
            yield json.dumps({"token": next_token, "text": tokenizer.decode([next_token])}) + "\n"
            tokens.append(next_token)
            time.sleep(0.05)
        # add EOS marker
        yield json.dumps({"token": None, "finish_reason": "eos"}) + "\n"
    return StreamingResponse(chat_stream(), media_type="application/json")

@router.post("/stream_chat_beam")
def stream_chat_beam(req: BeamStreamRequest):
    def beam_stream() -> Iterator[str]:
        context_text = " ".join(req.messages)
        prompt_ids = tokenizer.encode(context_text)
        beams = []
        for i in range(req.beam_width):
            output_ids = []
            decoder.generate(prompt_ids, output_ids, req.max_tokens, req.temperature)
            beams.append(tokenizer.decode(output_ids))

        if req.use_rerank:
            idx = reranker.select_best(context_text, beams)
        else:
            idx = 0

        best_tokens = tokenizer.encode(beams[idx])
        for token in best_tokens:
            yield json.dumps({"token": token, "text": tokenizer.decode([token])}) + "\n"
            time.sleep(0.03)
        # add EOS marker
        yield json.dumps({"token": None, "finish_reason": "eos"}) + "\n"
    return StreamingResponse(beam_stream(), media_type="application/json")
