import torch
import os
from models import PaliGemmaProcessor, PaliGemma, KVCache
from transformers import GemmaTokenizerFast
from PIL import Image
from typing import List
from tqdm.auto import tqdm
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _select_top_p(probs, top_p):
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    mask = cumsum_probs - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs.div_(torch.sum(sorted_probs, -1, keepdim=True))
    
    next_tokens = torch.multinomial(sorted_probs, num_samples=1)
    
    next_tokens = torch.gather(sorted_ids, dim=-1, index=next_tokens)
    
    return next_tokens
    
def inference(model: PaliGemma,
              processor: PaliGemmaProcessor,
              input_imgs: List[Image.Image],
              input_prompts: List[str],
              max_gen_len: int,
              temperature: float = 0.6,
              top_p: float = 0.9,
              device: str = 'cpu'):
    
    model = model.to(device=device).eval()
    
    
    model_inputs = processor(input_imgs,
                             input_prompts)
    
    # (n, seq_len)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    batch_size, _ = input_ids.shape
    processed_imgs = model_inputs["pixel_values"]
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    kv_cache = KVCache()
    
    max_prompt_len = 0
    prompt_len = []
    prompt_mask = (input_ids != tokenizer.pad_token_id) & (input_ids != model.cfg.image_token_index)
    for msk in prompt_mask:
        prompt_len.append(torch.sum(msk).item())
        max_prompt_len = max(max_prompt_len, len(msk))
        
    total_len = max_prompt_len + max_gen_len
    
    pad_id = tokenizer.pad_token_id
    eos_token = tokenizer.eos_token_id
    tokens = torch.full((batch_size, total_len), pad_id, dtype=input_ids.dtype, device=input_ids.device)
    
    for i, toks in enumerate(input_ids):
        tokens[i, :len(toks)] = toks

    eos_reached = torch.tensor([False] * batch_size, device=device)
    mask_tokens = tokens != pad_id # True if token is input_token, False if token is pad_token
    
    input_tokens = tokens[:, :model.vision_tower.cfg.num_image_tokens]
    
    output_tokens = []
    with torch.no_grad():
        # Encode images
        img_embeds = model.vision_tower(processed_imgs)
        img_embeds = model.multi_modal_projector(img_embeds)
        for cur_pos in tqdm(range(model.vision_tower.cfg.num_image_tokens, total_len)):
            logits, kv_cache = model.forward(img_embeds=img_embeds,
                                            input_tokens=input_tokens,
                                            attention_mask=attention_mask[:, :cur_pos],
                                            kv_cache=kv_cache)
            logits = model.language_model.lm_head(logits)
            
            kv_cache = kv_cache
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_tokens = _select_top_p(probs, top_p=top_p)
            else:
                next_tokens = torch.argmax(torch.softmax(logits[:, -1], dim=-1), dim=-1, keepdim=True)
            
            next_tokens = next_tokens.squeeze(-1)
            next_tokens = torch.where(mask_tokens[:, cur_pos], tokens[:, cur_pos], next_tokens)
        
            tokens[:, cur_pos] = next_tokens
            input_tokens = next_tokens.unsqueeze(-1)
            if cur_pos == attention_mask.shape[-1]:
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device, dtype=torch.int64)], dim=-1)
    
            eos_reached |= (~mask_tokens[:, cur_pos]) & (next_tokens == eos_token)
            
            if all(eos_reached):
                break
    
    output_tokens = []
    output_text = []
    
    for i, tok in enumerate(tokens.tolist()):
        if eos_token in tok:
            tok_index = tok.index(eos_token)
            tok = tok[:tok_index]
            
        output_tokens.append(tok[model.vision_tower.cfg.num_image_tokens:])
        output_text.append(tokenizer.decode(tok[model.vision_tower.cfg.num_image_tokens + prompt_len[i]:], skip_special_tokens=True))
    
    return output_tokens, output_text
        

if __name__ == '__main__':
    model = PaliGemma.from_pretrained(model_dir='./pretrained/model-3b-mix')
    tokenizer = load_tokenizer(tokenizer_dir='./pretrained/model-3b-mix')
    processor = PaliGemmaProcessor(tokenizer=tokenizer).from_pretrained(pretrained_dir='./pretrained/model-3b-mix')
    img = Image.open('./imgs/poster.jpg')
    img2 = Image.open('./imgs/dogandcat.jpeg')
    
    output_tokens, output_text = inference(model=model,
                                           processor=processor,
                                           input_imgs=[img2],
                                           input_prompts=['What are animals in the photo?'],
                                           max_gen_len=64,
                                           device='cpu',
                                           temperature=0.0)