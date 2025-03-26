# Tokenizer wrapper
# api/tokenizer.py
from transformers import AutoTokenizer
from typing import List, Tuple, Dict
from functools import lru_cache
import threading

class Tokenizer:
    _tokenizer_cache: Dict[str, "Tokenizer"] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def get(cls, model_name: str = "gpt2") -> "Tokenizer":
        with cls._lock:
            if model_name not in cls._tokenizer_cache:
                cls._tokenizer_cache[model_name] = Tokenizer(model_name)
            return cls._tokenizer_cache[model_name]

    @lru_cache(maxsize=4096)
    def encode_cached(self, text: str) -> Tuple[int]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> List[int]:
        return list(self.encode_cached(text))

    @lru_cache(maxsize=4096)
    def decode_cached(self, token_ids: Tuple[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def decode(self, token_ids: List[int]) -> str:
        return self.decode_cached(tuple(token_ids))

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def language(self) -> str:
        return self.model_name
