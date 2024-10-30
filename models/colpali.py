import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .gemma import KVCache
from .paligemma import PaliGemma, PaliGemmaConfig
from typing import Optional
from utils import *
from pathlib import Path
from safetensors import safe_open

def convert_weights_dict(original_weights):
    converted_weights = {}
    converted_weights['custom_text_proj.lora_A.weight'] = original_weights['base_model.model.custom_text_proj.lora_A.weight']
    converted_weights['custom_text_proj.lora_B.weight'] = original_weights['base_model.model.custom_text_proj.lora_B.weight']
    for i in range(18):
        converted_weights[f'model.language_model.model.layers.{i}.mlp.down_proj.lora_A.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.mlp.down_proj.lora_A.weight']
        converted_weights[f'model.language_model.model.layers.{i}.mlp.down_proj.lora_B.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.mlp.down_proj.lora_B.weight']
        converted_weights[f'model.language_model.model.layers.{i}.mlp.gate_proj.lora_A.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.mlp.gate_proj.lora_A.weight']
        converted_weights[f'model.language_model.model.layers.{i}.mlp.gate_proj.lora_B.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.mlp.gate_proj.lora_B.weight']
        converted_weights[f'model.language_model.model.layers.{i}.mlp.up_proj.lora_A.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.mlp.up_proj.lora_A.weight']
        converted_weights[f'model.language_model.model.layers.{i}.mlp.up_proj.lora_B.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.mlp.up_proj.lora_B.weight']
        converted_weights[f'model.language_model.model.layers.{i}.self_attn.q_proj.lora_A.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.self_attn.q_proj.lora_A.weight']
        converted_weights[f'model.language_model.model.layers.{i}.self_attn.q_proj.lora_B.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.self_attn.q_proj.lora_B.weight']
        converted_weights[f'model.language_model.model.layers.{i}.self_attn.k_proj.lora_A.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.self_attn.k_proj.lora_A.weight']
        converted_weights[f'model.language_model.model.layers.{i}.self_attn.k_proj.lora_B.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.self_attn.k_proj.lora_B.weight']
        converted_weights[f'model.language_model.model.layers.{i}.self_attn.v_proj.lora_A.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.self_attn.v_proj.lora_A.weight']
        converted_weights[f'model.language_model.model.layers.{i}.self_attn.v_proj.lora_B.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.self_attn.v_proj.lora_B.weight']
        converted_weights[f'model.language_model.model.layers.{i}.self_attn.o_proj.lora_A.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.self_attn.o_proj.lora_A.weight']
        converted_weights[f'model.language_model.model.layers.{i}.self_attn.o_proj.lora_B.weight'] = original_weights[f'base_model.model.model.language_model.model.layers.{i}.self_attn.o_proj.lora_B.weight']
    
    return converted_weights
        
        
class ColPali(nn.Module):
    def __init__(self, cfg: PaliGemmaConfig):
        super().__init__()
        self.model = PaliGemma(cfg=cfg)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.cfg.text_config.hidden_size, self.dim, bias=False)

    @staticmethod
    def from_pretrained(model_dir, torch_dtype: torch.dtype = torch.float32):
        torch.set_default_dtype(torch_dtype)
        with open(os.path.join(model_dir, 'config.json'), "r") as f:
            model_config = json.loads(f.read())
        config = PaliGemmaConfig.from_dict(model_config)
    
        safetensor_files = Path(model_dir).glob("*.safetensors")
        
        weights = {}
        for file in safetensor_files:
            with safe_open(file, framework='pt', device="cpu") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        model = ColPali(config)
        model.load_state_dict(weights, strict=False)
        model.tie_weights()
        return model
    
    def load_lora(self, model_dir):
        weights = {}
        with safe_open(os.path.join(model_dir, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
        converted_weights = convert_weights_dict(weights)
        self.load_state_dict(converted_weights, strict=False)
        
    def tie_weights(self):
        self.model.language_model.tie_weights()
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        outputs = self.model(*args, **kwargs)
        last_hidden_states = outputs[0]
        proj = self.custom_text_proj(last_hidden_states)
        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        proj = proj * kwargs['attention_mask'].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return proj
        
        
                
        
    