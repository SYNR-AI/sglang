from typing import Any, List
from pydantic import BaseModel


class Config(BaseModel):
    _class_name: str = "WanModel"
    _diffusers_version: str = "0.30.0"
    dim: int = 5120 # hidden_size
    eps: float = 1e-06
    ffn_dim: int = 13824
    freq_dim: int = 256
    in_dim: int = 16
    model_type: str = "t2v"
    num_heads: int = 40 # num_attention_heads
    num_layers: int = 40 # num_hidden_layers
    out_dim: int = 16
    text_len: int = 512

    architectures: List[str] = []

    config: Any = None
    checkpoint_dir: str = ""
    device_id: int = None
    rank: int = None
    t5_fsdp: bool = False
    dit_fsdp: bool = False
    use_usp: bool = False
    t5_cpu: bool = False
