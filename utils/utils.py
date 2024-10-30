import torch
from PIL import Image
from typing import Tuple, List
import numpy as np
import torch.nn as nn
import os
from transformers import AutoTokenizer, GemmaTokenizerFast
from safetensors import safe_open
import json
from pathlib import Path
from models.paligemma import PaliGemmaConfig, PaliGemma


def load_model(model_dir: str):
    
    with open(os.path.join(model_dir, 'config.json'), "r") as f:
        model_config = json.loads(f.read())
        config = PaliGemmaConfig.from_dict(model_config)
    
    safetensor_files = Path(model_dir).glob("*.safetensors")
    
    weights = {}
    for file in safetensor_files:
        with safe_open(file, framework='pt', device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
    model = PaliGemma(config)
    model.load_state_dict(weights, strict=False)
    model.tie_weights()
    
    return model


def load_tokenizer(tokenizer_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, padding_side='right')
    return tokenizer


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    
    return model