import torch
from PIL import Image
from typing import Tuple, List
import numpy as np
from transformers import GemmaTokenizerFast, BatchFeature
import json
import os

def preprocess_imgs(imgs: List[Image.Image],
                img_size: Tuple[int, int],
                rescale: float,
                mean: Tuple[float, float, float],
                std: Tuple[float, float, float]):
    
    def normalize(img, mean, std):
        img = (img - np.array(mean, dtype=img.dtype)) / np.array(std, dtype=img.dtype)
        return img
        
    resized_imgs = [np.array(img.resize((img_size[0], img_size[1]), resample=3)) for img in imgs]
    
    rescaled_imgs = [(img * rescale).astype(np.float32) for img in resized_imgs]
    
    
    normalized_imgs = [normalize(img, mean, std) for img in rescaled_imgs]
    transposed_imgs = [img.transpose(2, 0, 1) for img in normalized_imgs]
    
    tensor_imgs = torch.tensor(np.stack(transposed_imgs, axis=0), dtype=torch.float32)
    return tensor_imgs
    
    
def preprocess_prompts(prompt, image_token, max_num_image_token, bos_token):
    return f"{image_token * max_num_image_token}{bos_token}{prompt}\n"


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"
    def __init__(self,
                 tokenizer: GemmaTokenizerFast) -> None:
        
        additional_special_tokens = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(additional_special_tokens)
        
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]
        
        tokenizer.add_tokens(EXTRA_TOKENS)
        
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        self.tokenizer = tokenizer
    
    def from_pretrained(self, pretrained_dir):
        
        with open(os.path.join(pretrained_dir, "preprocessor_config.json"), "r") as f:
            config = json.loads(f.read())
        
        self.image_seq_length = config['image_seq_length']
        self.image_mean = config['image_mean']
        self.image_std = config['image_std']
        self.resample = config['resample']
        self.rescale_factor = config['rescale_factor']
        self.size = (config['size']['height'], config['size']['width'])
        return self
        
    
    def __call__(self,
                 imgs: List[Image.Image],
                 prompts: List[str],
                 padding: str = "longest", 
                 truncation: bool = True,
                 max_length: int = None):
        
        processed_imgs = preprocess_imgs(imgs,
                                      img_size=self.size,
                                      rescale=self.rescale_factor,
                                      mean=self.image_mean,
                                      std=self.image_mean)
        
        processed_prompts =  [preprocess_prompts(prompt, 
                                          image_token=self.IMAGE_TOKEN, 
                                          max_num_image_token=self.image_seq_length, 
                                          bos_token=self.tokenizer.bos_token) for prompt in prompts]
    
        model_inputs = self.tokenizer(processed_prompts,
                                    return_tensors='pt',
                                    padding=padding,
                                    truncation=truncation,
                                    max_length=max_length)
        
        return {**model_inputs, "pixel_values": processed_imgs}

        
        
        
        

        
        