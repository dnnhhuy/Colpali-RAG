import os
import torch
import base64
import asyncio
from io import BytesIO
import gradio as gr
import qdrant_client
from PIL import Image
from typing import List, Dict, Tuple

import llamaindex_utils
from rag_pipeline import async_indexDocument
from models import get_lora_model, enable_lora, ColPali, ColPaliProcessor
from utils import load_tokenizer

from llama_index.llms.gemini import Gemini
from llama_index.core.tools import RetrieverTool
from huggingface_hub import hf_hub_download

GEMINI_API_KEY = os.getenv(key="GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv(key="QDRANT_API_KEY")
HF_TOKEN_KEY = os.getenv(key="HF_TOKEN_KEY")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

async def initialize_model() -> Dict:
    """Initialize models

    Returns:
        model_dict: Dict: Dictionary stores neccessary models
    """
    if not os.path.exists("./pretrained/colpaligemma-3b-mix-448-base"):
        os.makedirs("./pretrained/colpaligemma-3b-mix-448-base", exist_ok=True)
        files_to_download = ["adapter_model.safetensors",
                            "config.json",
                            "model-00001-of-00002.safetensors",
                            "model-00002-of-00002.safetensors",
                            "preprocessor_config.json",
                            "tokenizer.json",
                            "tokenizer.model",
                            "tokenizer_config.json"]
        for file in files_to_download:
            hf_hub_download(repo_id="dnnhhuy/colpaligemma-3b-mix-448-base",
                            filename=file,
                            token=HF_TOKEN_KEY,
                            local_dir="./pretrained/colpaligemma-3b-mix-448-base")
        
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
    
    embed_model = llamaindex_utils.ColPaliGemmaEmbedding(model=model,
                                                         processor=processor,
                                                         device=device)
    
    collections = await get_collection_names(vector_store_client)
    retrievers_dict = {}
    for name in collections:
        if name not in retrievers_dict:
            retrievers_dict[name] = llamaindex_utils.ColPaliRetriever(vector_store_client=vector_store_client,
                                                                    target_collection=name,
                                                                    embed_model=embed_model,
                                                                    similarity_top_k=3)
    return {"llm": llm,
            "vector_store_client": vector_store_client,
            "model": model,
            "processor": processor,
            "embed_model": embed_model,
            "collections": collections,
            "retrievers_dict": retrievers_dict}

async def get_collection_names(vector_store_client):
    collections = await vector_store_client.get_collections()
    return [collection.name for collection in collections.collections]

async def index(files: List[str], 
          target_collection: str
          ) -> Tuple[str, gr.Dropdown, List[str], Dict[str, llamaindex_utils.ColPaliRetriever]]:
    """
    Insert all image pages from files to speicified target collection to the vector store
    and return the mapping from retriever's name to its object instance.

    Args:
        files (List[str]): List of file path
        target_collection (str): Target collection to insert into the vector store
        
    Returns:
        Tuple[str, gr.Dropdown, List[str], Dict[str, llamaindex_utils.ColPaliRetriever]]: Return message, dropdown component, collections' names, dictionary mapping retriever to its object instance
    """
    
    for file in files:
        await async_indexDocument(file_path=file,
                                vector_store_client=model_dict["vector_store_client"],
                                target_collection=target_collection,
                                model=model_dict["model"],
                                processor=model_dict["processor"],
                                device=device)
    
    if target_collection not in retrievers:
        retrievers[target_collection] = llamaindex_utils.ColPaliRetriever(vector_store_client=model_dict["vector_store_client"],
                                                                            target_collection=target_collection,
                                                                            embed_model=model_dict["embed_model"],
                                                                            similarity_top_k=3)
    collection_names = await get_collection_names(model_dict["vector_store_client"])
    return (f"Uploaded and index {len(files)} files.",
            gr.Dropdown(choices=collection_names), 
            collection_names)

async def search_with_llm(query: str,
                    similarity_top_k: int,
                    num_children: int) -> Tuple[str, List[Image.Image]]:
    """Search the result given query and list of retrievers.
    Returns the search's response and list of images support for that response.

    Args:
        query (str): Query question
        retrievers (Dict[str, llamaindex_utils.ColPaliRetriever]): Dictionary mapping between retrievers' names and their object instances
        similarity_top_k (int): top K similarity results retrieved from the retriever
        num_children (int): number of children for tree summarization

    Returns:
        Tuple[str, List[Image.Image]]:  Returns the search's response and list of images support for that response.
    """
    retriever_tools = [RetrieverTool.from_defaults(
                        name=key,
                        retriever=value,
                        description=f"Useful for retrieving information about {key} financials") for key, value in retrievers.items()]
    
    retriever_mappings = {retriever_tool.metadata.name: retriever_tool.retriever for retriever_tool in retriever_tools}
    
    fusion_retriever = llamaindex_utils.CustomFusionRetriever(llm=model_dict["llm"],
                                                            retriever_mappings=retriever_mappings,
                                                            similarity_top_k=similarity_top_k)
    
    query_engine = llamaindex_utils.CustomQueryEngine(retriever_tools=[retriever_tool.metadata for retriever_tool in retriever_tools],
                                                    fusion_retriever=fusion_retriever,
                                                    llm=model_dict["llm"],
                                                    num_children=num_children)
    response = await query_engine.aquery(query_str=query)
    
    return response.response, [Image.open(BytesIO(base64.b64decode(image))) for image in response.source_images]

async def delete_collection(target_collection):
    if await model_dict["vector_store_client"].collection_exists(collection_name=target_collection):
        await model_dict["vector_store_client"].delete_collection(collection_name=target_collection, timeout=100)
        choices = await get_collection_names(model_dict["vector_store_client"])
        return (f"Deleted collection {target_collection}", gr.Dropdown(choices=choices), choices)
    else:
        choices = await get_collection_names(model_dict["vector_store_client"])
        return (f"Collection {target_collection} is not found.", gr.Dropdown(choices=choices), choices)
        


def build_gui():
    with gr.Blocks() as demo:
        gr.Markdown("# Image Based RAG System using ColPali 📚🔍")
        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("## 1️. Upload PDFs")
                files = gr.File(file_types=["pdf"], 
                                file_count="multiple", 
                                interactive=True)
                
                choices = gr.State(value=model_dict["collections"])
                gr.Markdown("## 2️. Index the PDFs and upload")
                target_collection = gr.Dropdown(choices=choices.value, 
                                                allow_custom_value=True,
                                                label="Collection name", 
                                                show_label=True,
                                                interactive=True)
                
                message_box = gr.Textbox(value="File not yet uploaded", 
                                        show_label=False,
                                        interactive=False)
                with gr.Row(equal_height=True):
                    delete_button = gr.Button("🗑️ Delete collection")
                    convert_button = gr.Button("🔄 Convert and upload")
                
                # Define the actions for conversion
                convert_button.click(index, inputs=[files, target_collection], outputs=[message_box, target_collection, choices])
                
                # Define the actions for delete collection
                delete_button.click(delete_collection, inputs=[target_collection], outputs=[message_box, target_collection, choices])
                    
            with gr.Column():
                gr.Markdown("## 3️. Enter your question")
                query = gr.Textbox(placeholder="Enter your query to match",
                                lines=15,
                                max_lines=20,
                                autoscroll=True)
                with gr.Accordion(label="Additional Settings", open=False):
                    similarity_top_k = gr.Slider(minimum=1,
                                                maximum=10,
                                                value=3,
                                                step=1.0,
                                                label="Top K similarity retrieved from the retriever")
                    
                    num_children = gr.Slider(minimum=1, 
                                            maximum=10,
                                            value=3,
                                            step=1.0,
                                            label="Set number of children for Tree Summarization")
                search_button = gr.Button("🔍 Search")
                
        gr.Markdown("## 4️. ColPali Retrieval")
        with gr.Row(equal_height=True):
            output_text = gr.Textbox(label="Query result",
                                    show_label=True,
                                    placeholder="Response from query",
                                    lines=8,
                                    max_lines=20,
                                    interactive=False)
            output_imgs = gr.Gallery(label="Most relevant images is...", 
                                        show_fullscreen_button=True, 
                                        show_label=True, 
                                        show_download_button=True,
                                        interactive=False)
            
                
        # Action for search button
        search_button.click(
                    search_with_llm,
                    inputs=[query, similarity_top_k, num_children],
                    outputs=[output_text, output_imgs])
    return demo

async def amain():
    global model_dict, retrievers
    model_dict = await initialize_model()
    retrievers = model_dict["retrievers_dict"]
    
    demo = build_gui()
    demo.queue().launch(debug=True, share=False)
    
    
if __name__ == "__main__":
    asyncio.run(amain())
    
    