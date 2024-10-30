import torch
from PIL import Image
from typing import Tuple, List
import numpy as np
from transformers import GemmaTokenizerFast
from .paligemma_processor import PaliGemmaProcessor
from typing import Optional

def process_imgs(imgs: List[Image.Image],
                img_size: Tuple[int, int],
                rescale: float,
                mean: Tuple[float, float, float],
                std: Tuple[float, float, float]):
    
    def normalize(img, mean, std):
        img = (img - np.array(mean, dtype=img.dtype)) / np.array(std, dtype=img.dtype)
        return img
        
    resized_imgs = [img.resize((img_size[0], img_size[1]), resample=Image.Resampling.BICUBIC) for img in imgs]
    
    rescaled_imgs = [np.array(img, dtype=np.float32) * rescale for img in resized_imgs]
    
    normalized_imgs = [normalize(img, mean, std) for img in rescaled_imgs]
    
    transposed_imgs = [img.transpose(2, 0, 1) for img in normalized_imgs]
    
    tensor_imgs = torch.tensor(np.stack(transposed_imgs, axis=0), dtype=torch.float32)
    return tensor_imgs
    
    
def process_prompts(prompt, image_token, max_num_image_token, bos_token):
    return f"{image_token * max_num_image_token}{bos_token}{prompt}\n"


class ColPaliProcessor(PaliGemmaProcessor):
    def __init__(self,
                 tokenizer: GemmaTokenizerFast) -> None:
        super().__init__(tokenizer=tokenizer) 
        self.mock_image = Image.new(mode='RGB', size=(16, 16), color='black')
        
    def process_images(self, images: List[Image.Image]):
        input_prompts = ["Describe the image."] * len(images)
        
        images = [image.convert("RGB") for image in images]
        
        return_data = self(images,
                           input_prompts,
                           padding="longest",
                           truncation=False)
        
        return return_data
    
    def process_queries(self, 
                        queries: List[str],
                        max_length: int = 50,
                        suffix: Optional[str] = None):

        if suffix is None:
            suffix = "<pad>" * 10
        
        texts_query: List[str] = []
        
        for query in queries:
            query = f"Question: {query}"
            query += suffix
            texts_query.append(query)
            

        batch_query = self(imgs=[self.mock_image] * len(texts_query),
                            prompts=texts_query,
                            padding="longest",
                            max_length=max_length + self.image_seq_length,
                            truncation=True)

        del batch_query["pixel_values"]
        
        batch_query["input_ids"] = batch_query["input_ids"][..., self.image_seq_length:]
        batch_query["attention_mask"] = batch_query["attention_mask"][..., self.image_seq_length:]
        
        return batch_query
        
        
        
        
        
        

        
        