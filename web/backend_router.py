# web/backend_router.py
from decoder.cuda_decoder import CUDADecoder
from decoder.int8_decoder import INT8Decoder
from api.tokenizer import Tokenizer
from web.config import BACKEND_MODE

tokenizer = Tokenizer.get("gpt2")

decoder_gpu = CUDADecoder(12, 12, 64, 768, 50257, 2048)
decoder_gpu.load_weights("weights")
decoder_cpu = INT8Decoder(12, 12, 64, 768, 50257, 2048)
decoder_cpu.load_quantized_weights("weights/int8")

def select_backend():
    if BACKEND_MODE == "gpu":
        return decoder_gpu
    elif BACKEND_MODE == "cpu":
        return decoder_cpu
    elif BACKEND_MODE == "auto":
        # Todo: hybrid decoder of cpu and gpu
        return decoder_cpu
    else:
        raise decoder_cpu

def get_tokenizer():
    return tokenizer
