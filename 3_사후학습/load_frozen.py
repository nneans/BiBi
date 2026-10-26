"""Load the selected Stage 1 encoder with fixed weights and deterministic dropout."""
from pathlib import Path
from transformers import AutoModel
from stage1_core import FrozenEncoder


def load_encoder(device='cuda'):
    path = Path(__file__).resolve().parent / 'frozen-encoder'
    model = AutoModel.from_pretrained(str(path), local_files_only=True, use_safetensors=True,
                                      attn_implementation='sdpa')
    return FrozenEncoder(model).to(device)
