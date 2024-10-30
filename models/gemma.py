import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from dataclasses import dataclass
from typing import Optional, List
import math
import torch.utils.checkpoint as checkpoint

@dataclass
class GemmaConfig:
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_attention_heads: int = 8
    num_hidden_layers: int = 18
    num_image_tokens: int = 256
    num_key_value_heads: int = 1
    vocab_size: int = 257216
    norm_eps: float = 1e-6
    max_seq_len: int = 8192
    attention_dropout: float = 0.0
    use_lora: bool = False
    training: bool = False
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            hidden_size = data['hidden_size'],
            intermediate_size = data['intermediate_size'],
            num_attention_heads = data['num_attention_heads'],
            num_hidden_layers = data['num_hidden_layers'],
            num_image_tokens = data['num_image_tokens'],
            num_key_value_heads = data['num_key_value_heads'],
            vocab_size = data['vocab_size'],
            training = data['training'])

class RMSNorm(nn.Module):
    def __init__(self, dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.norm_eps = norm_eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)
    
    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def precompute_freqs(head_dim: int, max_seq_len: int, theta: int = 10000):
    thetas = 1 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    m = torch.arange(max_seq_len, dtype=torch.long)
    
    # (max_seq_len, head_dim // 2)
    freqs = torch.outer(m, thetas)
    
    # (max_seq_len, head_dim // 2) -> (max_seq_len, head_dim)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs

def roate_half(x: torch.Tensor):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_embed(x: torch.Tensor,
                       freqs: torch.Tensor):
    # x: (n, n_heads, seq_len, head_dim)
    # freqs: (n, seq_len, head_dim)
    device_type = x.device.type
    device_type = device_type if device_type != 'mps' else 'cpu'
    with torch.autocast(device_type=device_type, enabled=False):
        cos = freqs.cos()
        sin = freqs.sin()
        while len(cos.shape) < len(x.shape):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    x = (x * cos) + (roate_half(x) * sin)
    return x

class KVCache:
    def __init__(self):
        self.cache_k: List[torch.Tensor] = []
        self.cache_v: List[torch.Tensor] = []
    
    def num_items(self):
        if len(self.cache_k) == 0:
            return 0
        else:
            # (n, num_heads, seq_len, head_dim)
            return self.cache_k[0].shape[-2]
    
    def update(self, xk, xv, layer_idx):
        if layer_idx < len(self.cache_k):
            self.cache_k[layer_idx] = torch.cat((self.cache_k[layer_idx], xk), dim=-2)
            self.cache_v[layer_idx] = torch.cat((self.cache_v[layer_idx], xv), dim=-2)
        else:
            self.cache_k.append(xk)
            self.cache_v.append(xv)
        
        return self.cache_k[layer_idx], self.cache_v[layer_idx]
        
    
class GemmaTransformerAttention(nn.Module):
    def __init__(self, cfg: GemmaConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        self.num_key_value_heads = cfg.num_key_value_heads
        self.max_seq_len = cfg.max_seq_len
        
        assert self.hidden_size % self.num_attention_heads == 0
        
        self.n_rep =self.num_attention_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.attn_dropout = cfg.attention_dropout
        self.training = cfg.training

        self.register_buffer('freqs',
                      precompute_freqs(self.head_dim, cfg.max_seq_len),
                      persistent=False)
  
    def forward(self, x: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None):
        batch_size, seq_len, embed_dim = x.shape
        
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        
        # (n, seq_len, hidden_size) -> (n, seq_len, num_heads, head_dim) -> (n, num_heads, seq_len, head_dim)
        xq = xq.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        # (n, seq_len, hidden_size) -> (n, seq_len, num_kv_heads, head_dim) -> (n, num_kv_heads, seq_len, head_dim)
        xk = xk.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        xq = apply_rotary_embed(xq, self.freqs[position_ids, :])
        xk = apply_rotary_embed(xk, self.freqs[position_ids, :])
        
        if kv_cache is not None:
            keys, values = kv_cache.update(xk, xv, self.layer_idx)
        else:
            keys, values = xk, xv
        
        # (n, num_kv_heads, seq_len, head_dim) -> (n, num_kv_heads * n_rep, seq_len, head_dim) -> (n, num_heads, seq_len, head_dim)
        keys = keys[:, :, None, :, :].expand(-1, -1, self.n_rep, -1, -1).view(batch_size, -1, keys.shape[-2], self.head_dim)
        values = values[:, :, None, :, :].expand(-1, -1, self.n_rep, -1, -1).view(batch_size, -1, keys.shape[-2], self.head_dim)
        
        assert attention_mask is not None
        # (n, num_heads, seq_len, head_dim) @ (n, num_heads, head_dim, seq_len) -> (n, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(xq @ keys.transpose(2, 3) / math.sqrt(self.head_dim) + attention_mask, dim=-1)
        
        # dropout when training
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        # (n, num_heads, seq_len, seq_len) @ (n, num_heads, seq_len, head_dim) -> (n, num_heads, seq_len, head_dim)
        attn_output = attn_weights @ values
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*x.shape)
        
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
        

class GemmaTransformerMLP(nn.Module):
    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        self.cfg = cfg
        
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
    
    def forward(self, x: torch.Tensor):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        
        
        
class GemmaTransformerDecoder(nn.Module):
    def __init__(self, cfg: GemmaConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.self_attn = GemmaTransformerAttention(cfg, layer_idx)
        self.mlp = GemmaTransformerMLP(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.gradient_checking = False
            
    
    def forward(self, x: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None):
        
        residual = x
        x = self.input_layernorm(x)
        
        if self.gradient_checking:
            x = checkpoint.checkpoint(self.self_attn, x, position_ids, attention_mask, kv_cache)
        else:
            x = self.self_attn(x,
                               position_ids,
                               attention_mask,
                               kv_cache)[0]
        x += residual
        
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x
        
        
class GemmaModel(nn.Module):
    def __init__(self, cfg: GemmaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        
        self.layers = nn.ModuleList(
            [GemmaTransformerDecoder(cfg, layer_idx) for layer_idx in range(cfg.num_hidden_layers)]
        )
        
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
    
    def forward(self, x: torch.Tensor,
                position_ids: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                kv_cache: Optional[KVCache]) -> torch.Tensor:
        
        output = x * torch.tensor(self.cfg.hidden_size ** 0.5, dtype=x.dtype)
        for layer in self.layers:
            output = layer(output,
                           position_ids,
                           attention_mask,
                           kv_cache)
        output = self.norm(output)
        return output
            
    
class Gemma(nn.Module):
    def __init__(self, cfg: GemmaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = GemmaModel(cfg)
        self.vocab_size = cfg.vocab_size
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
                
    
    def gradient_checkpointing_enabled(self, enabled=False):
        for name, module in self.model.named_modules():
            if isinstance(module, GemmaTransformerDecoder):
                module.gradient_checking = enabled
        
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self, 
                input_embeds: torch.Tensor,
                position_ids: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                kv_cache: Optional[KVCache]):
        
        output = self.model(input_embeds,
                            position_ids,
                            attention_mask,
                            kv_cache)
        return output, kv_cache