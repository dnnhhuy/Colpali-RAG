import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .gemma import GemmaConfig, Gemma, KVCache
from .siglip import SigLIPConfig, SigLIPVisionTower
from typing import Optional
import os
import json
from pathlib import Path
from safetensors import safe_open

@dataclass
class PaliGemmaConfig:
    bos_token_id: int = 2
    eos_token_id: int = 1
    hidden_size: int = 2048
    ignore_index: int = -100
    image_token_index: int = 257152
    pad_token_id: int = 0
    projection_dim: int = 2048
    text_config: GemmaConfig = None
    vision_config: SigLIPConfig = None
    vocab_size: int = 257216
    @classmethod
    def from_dict(cls, data):
        return cls(
            bos_token_id = data['bos_token_id'],
            eos_token_id = data['eos_token_id'],
            hidden_size = data['hidden_size'],
            ignore_index = data['ignore_index'],
            image_token_index = data['image_token_index'],
            pad_token_id = data['pad_token_id'],
            projection_dim = data['projection_dim'],
            text_config = GemmaConfig.from_dict(data['text_config']),
            vision_config = SigLIPConfig.from_dict(data['vision_config'])
        )

class PaliGemmaMultimodalProjector(nn.Module):
    def __init__(self, cfg: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(cfg.vision_config.hidden_size, cfg.vision_config.projection_dim)
    
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x
    
class PaliGemma(nn.Module):
    def __init__(self, cfg: PaliGemmaConfig):
        super().__init__()
        self.cfg = cfg
        self.language_model = Gemma(cfg.text_config)
        
        self.vision_tower = SigLIPVisionTower(cfg.vision_config)
        
        self.multi_modal_projector = PaliGemmaMultimodalProjector(cfg)
    
    def tie_weights(self):
        self.language_model.tie_weights()
        
    def _merge_img_embeds_and_input_embeds(self, img_embeds: torch.Tensor,
                                                input_embeds: torch.Tensor,
                                                input_tokens: torch.Tensor):
        batch_size, seq_len, embed_dim = input_embeds.shape
        scaled_img = img_embeds / (self.cfg.hidden_size ** 0.5)
        
        final_embeddings = torch.zeros((batch_size, seq_len, embed_dim), dtype=img_embeds.dtype, device=img_embeds.device)
        
        
        # (n, seq_len)
        text_mask = (input_tokens != self.cfg.pad_token_id) & (input_tokens != self.cfg.image_token_index)
        img_mask = input_tokens == self.cfg.image_token_index
        pad_mask = input_tokens == self.cfg.pad_token_id
        
        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        img_mask = img_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        
        # (n, seq_len, embed_dim)
        final_embeddings = torch.where(text_mask, input_embeds, final_embeddings)
        final_embeddings = final_embeddings.masked_scatter(img_mask, scaled_img)
        final_embeddings = torch.where(pad_mask, torch.zeros_like(final_embeddings), final_embeddings)
        
        return final_embeddings

    def _create_position_ids_and_attention_mask(self,
                                                device: str = '',
                                                dtype: torch.dtype = torch.float32,
                                                batch_size: int = 32,
                                                seq_len: int = 1,
                                                attention_mask: Optional[torch.Tensor] = None, 
                                                kv_cache: Optional[KVCache] = None):
        # Create Attention Mask
        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full((batch_size, seq_len, seq_len), 0, dtype=dtype, device=device)
            position_ids = attention_mask.cumsum(dim=-1).masked_fill_((attention_mask == 0), 1).to(device)
        
        else:
            assert seq_len == 1
            kv_len = kv_cache.num_items() + 1
            causal_mask = torch.full((batch_size, 1, kv_len), 0, dtype=dtype, device=device)
            position_ids = attention_mask.cumsum(dim=-1)[:, -1].to(device)
        
        # (n, seq_len, kv_len) -> (n, 1, seq_len, kv_len)
        causal_mask = causal_mask.unsqueeze(1)
        
        return position_ids, causal_mask

    @staticmethod
    def from_pretrained(model_dir):
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
        
        
    def forward(self, *args, **kwargs):
        
        # input_tokens: (n, seq_len)
        
        # -> (n, seq_len, embed_dim)
        kv_cache = kwargs['kv_cache'] if 'kv_cache' in kwargs else None
        input_tokens = kwargs['input_ids']
        pixel_values = kwargs['pixel_values'] if 'pixel_values' in kwargs else None
        attention_mask = kwargs['attention_mask']
        input_embeds = self.language_model.model.embed_tokens(input_tokens)
        if pixel_values is not None:
            img_embeds = self.vision_tower(pixel_values.to(input_embeds.dtype))
            img_embeds = self.multi_modal_projector(img_embeds)
            final_embeddings = self._merge_img_embeds_and_input_embeds(img_embeds=img_embeds,
                                                                        input_embeds=input_embeds,
                                                                        input_tokens=input_tokens)
        else:
            final_embeddings = input_embeds

        position_ids, causal_mask = self._create_position_ids_and_attention_mask(device=final_embeddings.device.type,
                                                                                    dtype=final_embeddings.dtype,
                                                                                    batch_size=final_embeddings.shape[0],
                                                                                    seq_len=final_embeddings.shape[1],
                                                                                    attention_mask=attention_mask,
                                                                                    kv_cache=kv_cache)
        
        outputs, kv_cache = self.language_model(
            input_embeds=final_embeddings,
            position_ids=position_ids,
            attention_mask=causal_mask,
            kv_cache=kv_cache
        )
        return outputs, kv_cache
        