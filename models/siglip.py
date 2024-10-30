import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class SigLIPConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    num_image_tokens: int = 256
    patch_size: int = 14
    projection_dim: int = 2048
    n_channels: int = 3
    img_size: int = 224
    norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            hidden_size = data['hidden_size'],
            intermediate_size = data['intermediate_size'],
            num_attention_heads = data['num_attention_heads'],
            num_hidden_layers = data['num_hidden_layers'],
            num_image_tokens = data['num_image_tokens'],
            patch_size = data['patch_size'],
            projection_dim = data['projection_dim']
        )

class SigLIPEmbedding(nn.Module):
    def __init__(self, cfg: SigLIPConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(cfg.n_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size, padding='valid')
        
        self.num_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.position_embedding = nn.Embedding(cfg.num_image_tokens, cfg.hidden_size)
        
        self.register_buffer('position_ids',
                             torch.arange(cfg.num_image_tokens).expand(1, -1),
                             persistent=False)
    
    def forward(self, x: torch.FloatTensor):
        # x: (n, c, h, w) -> (n, c, num_patch_h, num_patch_w)
        img_embeds = self.patch_embedding(x)
        # (n, c, num_patch_h, num_patch_w) -> (n, c, num_patches) -> (n, num_patches, c)
        img_embeds = img_embeds.reshape(*img_embeds.shape[:2], -1).transpose(1, 2)
        return img_embeds + self.position_embedding(self.position_ids.to(torch.int64))

class SigLIPTransformerAttention(nn.Module):
    def __init__(self, cfg: SigLIPConfig):
        super().__init__()
        self.cfg = cfg
        self.num_attention_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // self.num_attention_heads
        
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout_p = self.cfg.attention_dropout
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, num_patches, _ = x.shape
        
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        
        xq = xq.view(batch_size, num_patches, self.num_attention_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, num_patches, self.num_attention_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, num_patches, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # attn_weights = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
        
        # attn_output  = torch.matmul(attn_weights, xv)
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.view(batch_size, num_patches, -1)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p,
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_patches, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, None
           
class SigLIPTransformerMLP(nn.Module):
    def __init__(self, cfg: SigLIPConfig):
        super().__init__()
        self.cfg = cfg
        
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
    
    def forward(self, x: torch.Tensor):
        
        x = self.fc1(x)
        x = F.gelu(x, approximate='tanh')
        x = self.fc2(x)
        return x

class SigLIPTransformerBlock(nn.Module):
    def __init__(self, cfg: SigLIPConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)

        self.self_attn = SigLIPTransformerAttention(cfg)
        self.mlp = SigLIPTransformerMLP(cfg)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm1(x)
        x = residual + self.self_attn(x, attention_mask)[0]
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.mlp(x)
        return x
    
class SigLIPTransformerEncoder(nn.Module):
    def __init__(self, cfg: SigLIPConfig):
        super().__init__()
        
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [SigLIPTransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)]
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x
class SigLIPModel(nn.Module):
    def __init__(self, cfg: SigLIPConfig):
        super().__init__()
        self.embeddings = SigLIPEmbedding(cfg)
        self.encoder = SigLIPTransformerEncoder(cfg)
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)
    
    def forward(self, x: torch.Tensor):
        img_embed = self.embeddings(x)
        output = self.encoder(img_embed)
        output = self.post_layernorm(output)
        return output
        
    
        
class SigLIPVisionTower(nn.Module):
    def __init__(self, cfg: SigLIPConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_model = SigLIPModel(cfg)
    
    def forward(self, x: torch.Tensor):
        return self.vision_model(x)
        
        
        
