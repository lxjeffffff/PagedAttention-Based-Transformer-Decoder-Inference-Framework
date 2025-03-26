from pydantic import BaseModel
from typing import List, Optional

class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    input_ids: Optional[List[int]] = None
    max_tokens: int = 32
    temperature: float = 1.0

class GenerateResponse(BaseModel):
    output_ids: List[int]
    output_text: Optional[str] = None

class StreamChatRequest(BaseModel):
    messages: List[str]
    max_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 5
    top_p: float = 1.0

class BeamStreamRequest(BaseModel):
    messages: List[str]
    max_tokens: int = 64
    beam_width: int = 4
    temperature: float = 1.0
    use_rerank: bool = True
