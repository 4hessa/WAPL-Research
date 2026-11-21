"""Shared pieces for pretrained-LLM stages: Qwen2.5-0.5B with LoRA adapters on CPU."""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL_DIR = Path('models/qwen2.5-0.5b')
MODEL_SHA256 = '88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342'
MODEL_REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, generator):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=generator)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def load_model(seed, rank=8, alpha=16, targets=('q_proj', 'v_proj')):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    for layer in model.model.layers:
        for name in targets:
            setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name), rank, alpha, g))
    model.eval()  # no dropout in Qwen2; eval keeps determinism explicit
    return model


def adapter_params(model):
    return [p for n, p in model.named_parameters() if p.requires_grad]


def adapter_state(model):
    return [p.detach().clone() for p in adapter_params(model)]


def load_adapter_state(model, state):
    with torch.no_grad():
        for p, s in zip(adapter_params(model), state):
            p.copy_(s)


def token_chunks(path, tokenizer, length):
    ids = tokenizer(Path(path).read_text(encoding='utf-8'), add_special_tokens=False)['input_ids']
    full = len(ids) // length
    return torch.tensor(ids[:full * length], dtype=torch.long).reshape(full, length)


def lm_loss(model, batch, reduction='mean'):
    logits = model(input_ids=batch[:, :-1]).logits
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1),
                                       reduction='none').reshape(len(batch), -1).mean(dim=1)
    return loss.mean() if reduction == 'mean' else loss


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
