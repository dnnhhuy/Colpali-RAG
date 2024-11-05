import torch
import asyncio
from torch.utils.data import DataLoader
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
from pdf2image import pdf2image
from typing import List, Union
from tqdm.auto import tqdm

from utils import *
from models import ColPali, ColPaliProcessor, get_lora_model, enable_lora

import qdrant_client
from qdrant_client.http import models as rest
from llamaindex_utils import ColPaliGemmaEmbedding, ColPaliRetriever, CustomFusionRetriever, CustomQueryEngine
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import RetrieverTool

os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
def embed_imgs(model: ColPali,
               processor: ColPaliProcessor,
               input_imgs: List[Image.Image],
               device: str = 'cpu') -> List[torch.Tensor]:
    """Generates embeddings given images.

    Args:
        model (ColPali): Main model
        processor (ColPaliProcessor): Data Processor
        input_imgs (List[Image.Image]): List of input images
        device (str, optional): device to run model. Defaults to 'cpu'.

    Returns:
        List[torch.Tensor]: List of output embedings.
    """
    
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
    """Generate embeddings given queries.

    Args:
        model (ColPali): Embedding model
        processor (ColPaliProcessor): Data Processor
        queries (List[str]): List of query strings
        device (str, optional): Device to run model. Defaults to 'cpu'.

    Returns:
        List[torch.Tensor]: List of embeddings
    """
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
                        ps: List[torch.Tensor]) -> torch.FloatTensor:
    """Calculate similarity between 2 single vectors

    Args:
        qs (List[torch.Tensor]): First Embeddings
        ps (List[torch.Tensor]): Second Embeddings

    Returns:
        torch.FloatTensor: Score Tensor
    """
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
                        device: Union[torch.device|str] = "cpu") -> torch.FloatTensor:
    """Calculate MaxSim between 2 list of vectors.

    Args:
        qs (List[torch.Tensor]): List of query embeddings
        ps (List[torch.Tensor]): List of document embeddings
        batch_size (int, optional): Batch Size. Defaults to 8.
        device (Union[torch.device | str], optional): Device to cast tensor to. Defaults to "cpu".

    Returns:
        torch.FloatTensor: Score tensors.
    """

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
                  vector_store_client,
                  target_collection: str,
                  model: nn.Module,
                  processor: ColPaliProcessor,
                  device: Union[str|torch.device]) -> None:
    """Index document given file_path.
    Each page in document is embedded by ColPaliGemma Model, then insert into Qdrant vector store given target collection.
    Creates taret collection if it is not created in the vector store yet.

    Args:
        file_path (str): _description_
        vector_store_client (_type_): _description_
        target_collection (str): _description_
        model (nn.Module): _description_
        processor (ColPaliProcessor): _description_
        device (Union[str | torch.device]): _description_
    """
    document_images = []
    document_embeddings = []
    document_images.extend(pdf2image.convert_from_path(file_path))
            
    document_embeddings = embed_imgs(model=model,
                                     processor=processor,
                                     input_imgs=document_images,
                                     device=device)
    
    # Create Qdrant Collectioon
    if not vector_store_client.collection_exists(collection_name=target_collection):
        # Specify vectors_config
        scalar_quant = rest.ScalarQuantizationConfig(
            type=rest.ScalarType.INT8,
            quantile=0.99,
            always_ram=False
        )
        vector_params = rest.VectorParams(
            size=128,
            distance=rest.Distance.COSINE,
            multivector_config=rest.MultiVectorConfig(
                comparator=rest.MultiVectorComparator.MAX_SIM
            ),
            quantization_config=rest.ScalarQuantization(
                scalar=scalar_quant
            ),
        )
        vector_store_client.create_collection(
            collection_name=target_collection,
            on_disk_payload=True,
            optimizers_config=rest.OptimizersConfigDiff(
                indexing_threshold=100
            ),
            vectors_config=vector_params
        )

    # Add embedding to Qdrant Collection
    points = []
    for i, embedding in enumerate(document_embeddings):
        multivector = embedding.cpu().float().numpy().tolist()
        
        buffer = BytesIO()
        document_images[i].save(buffer, format='JPEG')
        image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        # Define payload
        payload = {}
        node_metadata = {"file_name": file_path,
                        "page_id": i + 1}
        
        node_content = {'id_': str(uuid.uuid5(uuid.NAMESPACE_OID, name=(file_path + str(i + 1)))),
                        'image': image_str,
                        "metadata": node_metadata}
        
        payload["_node_content"] = json.dumps(node_content)
        payload["_node_type"] = "ImageNode"

        # store ref doc id at top level to allow metadata filtering
        # kept for backwards compatibility, will consolidate in future
        payload["document_id"] = "None"  # for Chroma
        payload["doc_id"] = "None"  # for Pinecone, Qdrant, Redis
        payload["ref_doc_id"] = "None"  # for Weaviate
    
        points.append(rest.PointStruct(
            id=node_content["id_"],
            vector=multivector,
            payload=payload,
        ))
        
    step = 8
    for i in range(0, len(points), step):
        points_batch = points[i: i + step]
        vector_store_client.upsert(collection_name=target_collection,
                                points=points_batch,
                                wait=False)


async def async_indexDocument(file_path: str,
                  vector_store_client: qdrant_client.AsyncQdrantClient,
                  target_collection: str,
                  model: nn.Module,
                  processor: ColPaliProcessor,
                  device: Union[str|torch.device]) -> None:
    """Asynchrously index document given file_path.
    Each page in document is embedded by ColPaliGemma Model, then insert into Qdrant vector store given target collection.
    Creates taret collection if it is not created in the vector store yet.

    Args:
        file_path (str): _description_
        vector_store_client (_type_): _description_
        target_collection (str): _description_
        model (nn.Module): _description_
        processor (ColPaliProcessor): _description_
        device (Union[str | torch.device]): _description_
    """
    document_images = []
    document_embeddings = []
    document_images.extend(pdf2image.convert_from_path(file_path))
            
    document_embeddings = embed_imgs(model=model,
                                     processor=processor,
                                     input_imgs=document_images,
                                     device=device)
    
    # Create Qdrant Collectioon
    if not await vector_store_client.collection_exists(collection_name=target_collection):
        # Specify vectors_config
        scalar_quant = rest.ScalarQuantizationConfig(
            type=rest.ScalarType.INT8,
            quantile=0.99,
            always_ram=False
        )
        vector_params = rest.VectorParams(
            size=128,
            distance=rest.Distance.COSINE,
            multivector_config=rest.MultiVectorConfig(
                comparator=rest.MultiVectorComparator.MAX_SIM
            ),
            quantization_config=rest.ScalarQuantization(
                scalar=scalar_quant
            ),
        )
        await vector_store_client.create_collection(
            collection_name=target_collection,
            on_disk_payload=True,
            optimizers_config=rest.OptimizersConfigDiff(
                indexing_threshold=100
            ),
            vectors_config=vector_params
        )

    # Add embedding to Qdrant Collection
    points = []
    for i, embedding in enumerate(document_embeddings):
        multivector = embedding.cpu().float().numpy().tolist()
        
        buffer = BytesIO()
        document_images[i].save(buffer, format='JPEG')
        image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        # Define payload
        payload = {}
        node_metadata = {"file_name": file_path,
                        "page_id": i + 1}
        
        node_content = {'id_': str(uuid.uuid5(uuid.NAMESPACE_OID, name=(file_path + str(i + 1)))),
                        'image': image_str,
                        "metadata": node_metadata}
        
        payload["_node_content"] = json.dumps(node_content)
        payload["_node_type"] = "ImageNode"

        # store ref doc id at top level to allow metadata filtering
        # kept for backwards compatibility, will consolidate in future
        payload["document_id"] = "None"  # for Chroma
        payload["doc_id"] = "None"  # for Pinecone, Qdrant, Redis
        payload["ref_doc_id"] = "None"  # for Weaviate
    
        points.append(rest.PointStruct(
            id=node_content["id_"],
            vector=multivector,
            payload=payload,
        ))
    
    step = 8
    for i in range(0, len(points), step):
        points_batch = points[i: i + step]
        await vector_store_client.upsert(collection_name=target_collection,
                    points=points_batch,
                    wait=False)
  

GEMINI_API_KEY = os.getenv(key="GEMINI_API_KEY")

def main():
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
    
    # Initialize LLM
    generation_config = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
    }
    
    llm = Gemini(api_key=GEMINI_API_KEY, generation_config=generation_config)
    
    # Setup Qdrant
    # Creating Qdrant Client
    vector_store_client = qdrant_client.QdrantClient(location="http://localhost:6333", timeout=100)
    
    indexDocument('./data/pdfs-financial/Alphabet_Inc_goog-10-q-q1-2024.pdf',
                  vector_store_client=vector_store_client,
                  target_collection="Alphabet",
                  model=model, 
                  processor=processor, 
                  device='mps')
    
    indexDocument('./data/pdfs-financial/Nvidia_ecefb2b2-efcb-45f3-b72b-212d90fcd873.pdf',
                  vector_store_client=vector_store_client,
                  target_collection="Nvidia",
                    model=model, 
                    processor=processor, 
                    device='mps')
    
    # RAG using LLamaIndex 
    
    embed_model = ColPaliGemmaEmbedding(model=model, processor=processor, device="mps")
    
    alphabet_retriever = ColPaliRetriever(vector_store_client=vector_store_client,
                                          target_collection="Alphabet",
                                          embed_model=embed_model,
                                          query_mode='default',
                                          similarity_top_k=3)

    nvidia_retriever = ColPaliRetriever(vector_store_client=vector_store_client,
                                          target_collection="Nvidia",
                                          embed_model=embed_model,
                                          query_mode='default',
                                          similarity_top_k=3)
    
    # Query Router Among Multiple Retrievers
    retriever_tools = [
        RetrieverTool.from_defaults(
            name="alphabet",
            retriever=alphabet_retriever,
            description="Useful for retrieving information about Alphabet Inc financials"
            ),
        RetrieverTool.from_defaults(
            name="nvidia",
            retriever=nvidia_retriever,
            description="Useful for retrieving information about Nvidia financials"
            )
        ]
    
    retriever_mappings = {retriever_tool.metadata.name: retriever_tool.retriever for retriever_tool in retriever_tools}
    
    fusion_retriever = CustomFusionRetriever(llm=llm,
                                             retriever_mappings=retriever_mappings,
                                             num_generated_queries=3,
                                             similarity_top_k=3)
    
    query_engine = CustomQueryEngine(retriever_tools=[retriever_tool.metadata for retriever_tool in retriever_tools],
                                     fusion_retriever=fusion_retriever,
                                     llm=llm,
                                     num_children=3)
    
    query_str = "Compare the net income between Nvidia and Alphabet"
    response = query_engine.query(query_str=query_str)
    print(response.response)

async def amain():
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
    
    # Initialize LLM
    generation_config = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
    }
    
    llm = Gemini(api_key=GEMINI_API_KEY, generation_config=generation_config)
    
    # Setup Qdrant
    # Creating Qdrant Client
    vector_store_client = qdrant_client.AsyncQdrantClient(location="http://localhost:6333", timeout=100)
    
    await async_indexDocument('./data/pdfs-financial/Alphabet_Inc_goog-10-q-q1-2024.pdf',
                  vector_store_client=vector_store_client,
                  target_collection="Alphabet",
                  model=model, 
                  processor=processor, 
                  device='mps')
    
    await async_indexDocument('./data/pdfs-financial/Nvidia_ecefb2b2-efcb-45f3-b72b-212d90fcd873.pdf',
                                                    vector_store_client=vector_store_client,
                                                    target_collection="Nvidia",
                                                    model=model, 
                                                    processor=processor, 
                                                    device='mps')
    
    embed_model = ColPaliGemmaEmbedding(model=model, processor=processor, device="mps")
    
    alphabet_retriever = ColPaliRetriever(vector_store_client=vector_store_client,
                                          target_collection="Alphabet",
                                        embed_model=embed_model,
                                        query_mode='default',
                                        similarity_top_k=3)
    
    nvidia_retriever = ColPaliRetriever(vector_store_client=vector_store_client,
                                        target_collection="Nvidia",
                                        embed_model=embed_model,
                                        query_mode='default',
                                        similarity_top_k=3)
    
    
    # Query Router Among Multiple Retrievers
    retriever_tools = [
        RetrieverTool.from_defaults(
            name="alphabet",
            retriever=alphabet_retriever,
            description="Useful for retrieving information about Alphabet Inc financials"
            ),
        RetrieverTool.from_defaults(
            name="nvidia",
            retriever=nvidia_retriever,
            description="Useful for retrieving information about Nvidia financials"
            )
        ]
    
    retriever_mappings = {retriever_tool.metadata.name: retriever_tool.retriever for retriever_tool in retriever_tools}
    
    fusion_retriever = CustomFusionRetriever(llm=llm,
                                             retriever_mappings=retriever_mappings,
                                             similarity_top_k=3)
    
    query_engine = CustomQueryEngine(retriever_tools=[retriever_tool.metadata for retriever_tool in retriever_tools],
                                     fusion_retriever=fusion_retriever,
                                     llm=llm,
                                     num_children=3)
    
    query_str = "Compare the net income between Nvidia and Alphabet"
    response = await query_engine.aquery(query_str=query_str)
    print(str(response))

if __name__ == "__main__":
    main()