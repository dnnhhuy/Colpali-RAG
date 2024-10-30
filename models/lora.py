import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from typing import List

class LoRALayer:
    def __init__(self, features_in: int, features_out: int, rank: int=1, alphas: int=1):
        super().__init__()
        self.lora_A = nn.Linear(features_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, features_out, bias=False)
        nn.init.normal_(self.lora_A.weight, mean=0, std=1/rank)
        
        self.scale = alphas / rank

class LoRALinear(nn.Module, LoRALayer):
    def __init__(self, base_layer: nn.Module, rank: int=1, alphas: int=1, dropout_p: float=0.0):
        features_out, features_in = base_layer.weight.shape
        super().__init__()
        LoRALayer.__init__(self, features_in=features_in, features_out=features_out, rank=rank, alphas=alphas)
        
        self.base_layer = nn.Linear(features_in, features_out, bias=False)
        self.base_layer.weight = base_layer.weight
        
        if dropout_p > 0.0:
            self.lora_dropout = nn.Dropout(p=dropout_p, inplace=False)
        else:
            self.lora_dropout = nn.Identity()
            
        self.enabled = False
    
    def forward(self, x: torch.Tensor):
        result = self.base_layer(x)
        if self.enabled:
            result = result + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale
        return result
    
def enable_lora(model: nn.Module, lora_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], enabled=True):
    for name, module in model.named_modules():
       if name.split('.')[-1] in lora_modules:
           module.enabled = enabled
    return model

def replace_module(module: nn.Module, target_modules: List[str], torch_dtype: torch.dtype, **kwargs):
    for child_name, child_module in module.named_children():
        if child_name in target_modules:
            new_module = LoRALinear(child_module, **kwargs).to(torch_dtype)
            setattr(module, child_name, new_module)
        else:
            replace_module(child_module, target_modules, torch_dtype, **kwargs)
            
def get_lora_model(model: nn.Module, rank: float, alphas: float, lora_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], dropout_p: float = 0.0, training: bool = False, torch_dtype: torch.dtype = torch.bfloat16):
    lora_config = {'rank': rank,
                   'alphas': alphas,
                   'dropout_p': dropout_p}
    replace_module(model, lora_modules, torch_dtype, **lora_config)
           
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            if training:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    return model
    
