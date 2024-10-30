import torch
from torch.utils.data import DataLoader
import os
from models import ColPali, ColPaliProcessor, get_lora_model, enable_lora
from PIL import Image
from typing import List, Union, Optional, Any
from tqdm.auto import tqdm
from utils import *
from pdf2image import pdf2image

os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
def embed_imgs(model: ColPali,
               processor: ColPaliProcessor,
               input_imgs: List[Image.Image],
               device: str = 'cpu') -> List[torch.Tensor]:
    
    colpali_model = model.to(device=device).eval()

    dataloader = DataLoader(input_imgs,
                            batch_size=8,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=lambda x: processor.process_images(x))

    document_embeddings = []
    with torch.no_grad():
        for batch, model_inputs in tqdm(enumerate(dataloader)):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            # Encode images
            img_embeds = colpali_model(**model_inputs, kv_cache=None)
            document_embeddings.extend(list(torch.unbind(img_embeds.to('cpu').to(torch.float32))))
    return document_embeddings

def embed_queries(model: ColPali,
                  processor: ColPaliProcessor,
                  queries: List[str],
                  device: str = 'cpu') -> List[torch.Tensor]:
    colpali_model = model.to(device=device).eval()
    
    dataloader = DataLoader(queries,
                            batch_size=8,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=lambda x: processor.process_queries(x))
    
    queries_embeddings = []
    with torch.no_grad():
        for batch, model_inputs in tqdm(enumerate(dataloader)):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            # Encode Queries
            query_embeds = colpali_model(**model_inputs, kv_cache=None)
        queries_embeddings.extend(torch.unbind(query_embeds.to('cpu').type(torch.float32)))
        
    return queries_embeddings           


def score_single_vectors(qs: List[torch.Tensor], 
                        ps: List[torch.Tensor]):
    assert len(qs) != 0 and len(ps) != 0
    
    qs_stacked = torch.stack(qs)
    ps_stacked = torch.stack(ps)
    
    scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
    assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"
    scores = scores.to(torch.float32)
    return scores

def score_multi_vectors(qs: List[torch.Tensor],
                        ps: List[torch.Tensor],
                        batch_size: int = 8,
                        device: Union[torch.device|str] = "cpu"):

    assert len(qs) != 0 and len(ps) != 0
    scores_list = []
    for i in range(0, len(qs), batch_size):
        scores_batch = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i:i+batch_size], batch_first=True, padding_value=0).to(device)
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(ps[j:j+batch_size], batch_first=True, padding_value=0).to(device)
            tmp = torch.einsum("abd,ced->acbe", qs_batch, ps_batch).max(dim=-1)[0].sum(dim=2)
            scores_batch.append(tmp)
            
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores_list.append(scores_batch)
    
    scores = torch.cat(scores_list, dim=0)
    return scores.to(torch.float32)

def indexDocument(file_path: str,
                  model: nn.Module,
                  processor: ColPaliProcessor,
                  device: Union[str|torch.device]):
    document_images = []
    document_embeddings = []
    document_images.extend(pdf2image.convert_from_path(file_path))
            
    document_embeddings = embed_imgs(model=model,
                                     processor=processor,
                                     input_imgs=document_images[:10],
                                     device=device)
    
    
    return document_embeddings, document_images

if __name__ == '__main__':
    model = ColPali.from_pretrained(model_dir='./pretrained/colpaligemma-3b-mix-448-base', torch_dtype=torch.bfloat16)
    tokenizer = load_tokenizer(tokenizer_dir='./pretrained/colpaligemma-3b-mix-448-base')
    processor = ColPaliProcessor(tokenizer=tokenizer).from_pretrained(pretrained_dir='./pretrained/colpaligemma-3b-mix-448-base')
    
    model.model.language_model.model = get_lora_model(model.model.language_model.model, 
                                                      rank=32, 
                                                      alphas=32, 
                                                      lora_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj'], 
                                                      training=False,
                                                      dropout_p=0.1, 
                                                      torch_dtype=torch.bfloat16)
    model.model.language_model.model = enable_lora(model.model.language_model.model, lora_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj'], enabled=True)
    
    model = get_lora_model(model, 
                           rank=32, 
                           alphas=32, 
                           lora_modules=['custom_text_proj'], 
                           training=False, 
                           dropout_p=0.1, 
                           torch_dtype=torch.bfloat16)
    model = enable_lora(model, lora_modules=['custom_text_proj'], enabled=True)
    
    model.load_lora('./pretrained/colpaligemma-3b-mix-448-base')
    
    document_embeddings, document_images = indexDocument(file_path="",
                                                         model=model,
                                                         processor=processor,
                                                         device="mps")
   
    queries_embeddings = embed_queries(model=model,
                                       processor=processor,
                                       queries=[""],
                                       device="mps")
    
    max_ids = torch.argmax(score_multi_vectors(queries_embeddings, document_embeddings), dim=1)
    document_images[max_ids.item()].save('test.jpg')
    

