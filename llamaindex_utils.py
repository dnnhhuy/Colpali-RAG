import torch
import json
import asyncio
import qdrant_client
from PIL import Image
from pydantic import PrivateAttr, Field
from typing import Union, Optional, List, Any, Dict, Set
from dataclasses import dataclass

from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle, PromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.llms import LLM
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import ToolMetadata
from llama_index.core.output_parsers.utils import parse_json_markdown
from llama_index.core.question_gen.types import SubQuestion

from models import ColPali, ColPaliProcessor
from prompt_templates import (DEFAULT_GEN_PROMPT_TMPL, 
                     DEFAULT_FINAL_ANSWER_PROMPT_TMPL,
                     DEFAULT_SUB_QUESTION_PROMPT_TMPL,
                     DEFAULT_SYNTHESIZE_PROMPT_TMPL)
from typing import Any, List, Optional, Tuple, cast
from qdrant_client.http.models import Payload

from collections import defaultdict

def parse_to_query_result(response: List[Any]) -> VectorStoreQueryResult:
    """
    Convert vector store response to VectorStoreQueryResult.

    Args:
        response: List[Any]: List of results returned from the vector store.
    """
    nodes = []
    similarities = []
    ids = []

    for point in response:
        payload = cast(Payload, point.payload)
        try:
            node = metadata_dict_to_node(payload)
        except Exception:
            metadata, node_info, relationships = legacy_metadata_dict_to_node(
                payload
            )

            node = TextNode(
                id_=str(point.id),
                text=payload.get("text"),
                metadata=metadata,
                start_char_idx=node_info.get("start", None),
                end_char_idx=node_info.get("end", None),
                relationships=relationships,
            )
        nodes.append(node)
        ids.append(str(point.id))
        try:
            similarities.append(point.score)
        except AttributeError:
            # certain requests do not return a score
            similarities.append(1.0)

    return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
    

class ColPaliGemmaEmbedding(BaseEmbedding):
    _model: ColPali = PrivateAttr()
    _processor: ColPaliProcessor = PrivateAttr()
    
    device: Union[torch.device | str] = Field(default="cpu",
                                            description="Device to use")
    def __init__(self,
                model: ColPali,
                processor: ColPaliProcessor,
                device: Optional[str] = 'cpu',
                **kwargs):
        super().__init__(device=device,
                        **kwargs)
        self._model = model
        self._processor = processor
    
    @classmethod
    def class_name(cls) -> str:
        return "ColPaliGemmaEmbedding"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding.

        Args:
            query (str): Query String
        """
        
        processed_query = self._processor.process_queries([query])
        processed_query = {k: v.to(self.device) for k, v in processed_query.items()}
        query_embeddings = self._model(**processed_query)
        return query_embeddings.to('cpu')[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding.

        Args:
            text (str): Text String
        """
        
        processed_query = self._processor.process_queries([text])
        processed_query = {k: v.to(self.device) for k, v in processed_query.items()}
        query_embeddings = self._model(**processed_query)
        return query_embeddings.to('cpu')[0]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings.

        Args:
            texts (List[str]): List of text string
        """
        
        processed_queries = self._processor.process_queries(texts)
        processed_query = {k: v.to(self.device) for k, v in processed_query.items()}
        query_embeddings = self._model(**processed_queries)
        return query_embeddings.to('cpu')
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

class  ColPaliRetriever(BaseRetriever):
    def __init__(self,
                vector_store_client: Union[qdrant_client.QdrantClient | qdrant_client.AsyncQdrantClient],
                target_collection: str,
                embed_model: ColPaliGemmaEmbedding,
                query_mode: str = 'default',
                similarity_top_k: int = 2,
                ) -> None:
        self._vector_store_client = vector_store_client
        self._target_collection = target_collection
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Get retrived nodes from the vector store by retriever given query string.

        Args:
            query_bundle (QueryBundle): QueryBundle class includes query string

        Returns:
            List[NodeWithScore]: List of retrieved nodes.
        """
        if query_bundle.embedding is None:
            query_embedding = self._embed_model._get_query_embedding(query_bundle.query_str)
        else:
            query_embedding = query_bundle.embedding
            

        query_embedding = query_embedding.cpu().float().numpy().tolist()
        
        # Get nodes from vector store
        response = self._vector_store_client.search(collection_name=self._target_collection,
                                                    query_vector=query_embedding,
                                                    limit=self._similarity_top_k)
        
        # Parse to structured output nodes 
        query_result = parse_to_query_result(response)
        nodes_with_scores = []
        for idx, node in enumerate(query_result.nodes):
            score = None
            if query_result.similarities is not None:
                score = query_result.similarities[idx]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously get retrived nodes from the vector store by retriever given query string.

        Args:
            query_bundle (QueryBundle): QueryBundle class includes query string

        Returns:
            List[NodeWithScore]: List of retrieved nodes.
        """
        if query_bundle.embedding is None:
            query_embedding = await self._embed_model._aget_query_embedding(query_bundle.query_str)
        else:
            query_embedding = query_bundle.embedding
        
        query_embedding = query_embedding.cpu().float().numpy().tolist()
        
        # Get nodes from vector store
        response = await self._vector_store_client.search(collection_name=self._target_collection,
                                                            query_vector=query_embedding,
                                                            limit=self._similarity_top_k)
        
        # Parse to structured output nodes 
        query_result = parse_to_query_result(response)
        nodes_with_scores = []
        for idx, node in enumerate(query_result.nodes):
            score = None
            if query_result.similarities is not None:
                score = query_result.similarities[idx]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores
        

def fuse_results(retrieved_nodes: List[NodeWithScore], similarity_top_k: int) -> List[NodeWithScore]:
    """Fuse retrieved nodes using Reciprocal Rank

    Args:
        retrieved_nodes (List[NodeWithScore]): List of nodes.
        similarity_top_k (int): get top K nodes.

    Returns:
        List[NodeWithScore]: List of nodes after fused
    """
    k = 60.0
    fused_scores = {}
    text_to_node = {}
    for rank, node_with_score in enumerate(sorted(retrieved_nodes, key=lambda x: x.score or 0.0, reverse=True)):
        text = node_with_score.node.get_content(metadata_mode='all')
        text_to_node[text] = node_with_score
        fused_scores[text] = fused_scores.get(text, 0.0) + 1.0 / (rank + k)
        
    # Sort results by calculated score
    reranked_results = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score
    return reranked_nodes[:similarity_top_k]


def generate_queries(llm: LLM, query: str, num_queries: int) -> List[str]:
    """Generate num_queries queries

    Args:
        llm (LLM): LLM model
        query (str): query string
        num_queries (int): Number of queries to generate

    Returns:
        generate_queries List[str]: List of generated queries
    """
    query_prompt = PromptTemplate(DEFAULT_GEN_PROMPT_TMPL)
    generate_queries = llm.predict(query_prompt, 
                                    num_queries=num_queries, 
                                    query=query)
    generate_queries = generate_queries.split('\n')
    return generate_queries

async def agenerate_queries(llm: LLM, query: str, num_queries: int):
    """Asynchronously generate num_queries queries

    Args:
        llm (LLM): LLM model
        query (str): query string
        num_queries (int): Number of queries to generate

    Returns:
        generate_queries List[str]: List of generated queries
    """
    query_prompt = PromptTemplate(DEFAULT_GEN_PROMPT_TMPL)
    generate_queries = await llm.apredict(query_prompt, 
                                    num_queries=num_queries, 
                                    query=query)
    generate_queries = generate_queries.split('\n')
    return generate_queries


# Tree Summarization
def synthesize_results(queries: List[SubQuestion], contexts: Dict[str, Set[str]], llm: LLM, num_children: int) -> Tuple[str, List[str]]:
    """Summarize the results generated from LLM.

    Args:
        queries (List[SubQuestion]): Generated results
        contexts (Dict[str, Set[str]]): Dictionary maps context information string to its set of source images
        llm (LLM): LLM Model
        num_children (int): Number of children for Tree Summarization

    Returns:
        Tuple[str, List[str]]: Synthesized text, set of source images.
    """
    qa_prompt = PromptTemplate(DEFAULT_SYNTHESIZE_PROMPT_TMPL)
        
    new_contexts = defaultdict(set)
    keys = list(contexts.keys())
    for idx in range(0, len(keys), num_children):
        contexts_batch = keys[idx: idx + num_children]
        context_str = '\n\n'.join([f"{i + 1}. {text}" for i, text in enumerate(contexts_batch)])
        
        fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str="\n".join([query.sub_question for query in queries]))
        combined_result = llm.complete(fmt_qa_prompt)
        
        # Parse json string to dictionary
        json_dict = parse_json_markdown(str(combined_result))
        if len(json_dict['choices']) > 0:
            for choice in json_dict['choices']:
                new_contexts[json_dict['summarized_text']] = new_contexts[json_dict['summarized_text']].union(contexts[contexts_batch[choice - 1]])
        else:
            new_contexts[json_dict['summarized_text']] = set()
            
    if len(new_contexts) == 1:
        synthesized_text = list(new_contexts.keys())[0]
        return synthesized_text, list(new_contexts[synthesized_text])
    else:
        return synthesize_results(queries, new_contexts, llm, num_children=num_children)
    

async def asynthesize_results(queries: List[SubQuestion], contexts: Dict[str, Set[str]], llm: LLM, num_children: int) -> Union[str, List[str]]:
    """Asynchronously sumamarize the results generated from LLM.

    Args:
        queries (List[SubQuestion]): Generated results
        contexts (Dict[str, Set[str]]): Dictionary maps context information string to its set of source images
        llm (LLM): LLM Model
        num_children (int): Number of children for Tree Summarization

    Returns:
        Tuple[str, List[str]]: Synthesized text, set of source images.
    """
    qa_prompt = PromptTemplate(DEFAULT_SYNTHESIZE_PROMPT_TMPL)
    fmt_qa_prompts = []
    keys = list(contexts.keys())
    contexts_batches = []
    for idx in range(0, len(keys), num_children):
        contexts_batch = keys[idx: idx + num_children]
        
        context_str = '\n\n'.join([f"{idx + 1}. {text}" for idx, text in enumerate(contexts_batch)])
        
        fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str="\n".join([query.sub_question for query in queries]))
        fmt_qa_prompts.append(fmt_qa_prompt)
        contexts_batches.append(contexts_batch)
    
    tasks = []
    async with asyncio.TaskGroup() as tg:
        for fmt_qa_prompt in fmt_qa_prompts:
            task = tg.create_task(llm.acomplete(fmt_qa_prompt))
            tasks.append(task)
        
    responses = [str(task.result()) for task in tasks]
    new_contexts = defaultdict(set)
    for idx, response in enumerate(responses):
        # Parse json string to dictionary
        json_dict = parse_json_markdown(response)
        
        if len(json_dict["choices"]) > 1:
            for choice in json_dict["choices"]:
                new_contexts[json_dict["summarized_text"]] = new_contexts[json_dict["summarized_text"]].union(contexts[contexts_batches[idx][choice - 1]])
        else:
            new_contexts[json_dict["summarized_text"]] = set()
            
    if len(new_contexts) == 1:
        synthesized_text = list(new_contexts.keys())[0]
        return synthesized_text, list(new_contexts[synthesized_text])
    else:
        return await asynthesize_results(queries, new_contexts, llm, num_children=num_children)
    
class CustomFusionRetriever(BaseRetriever):
    def __init__(self,
                 llm,
                 retriever_mappings: Dict[str, BaseRetriever],
                 similarity_top_k: int = 3,
                 num_generated_queries = 3,
                 ) -> None:
        self._retriever_mappings = retriever_mappings
        self._similarity_top_k = similarity_top_k
        self._num_generated_queries = num_generated_queries
        self._llm = llm
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve self._similarity_top_k content nodes given query

        Args:
            query_bundle (QueryBundle): query bundle include query string
        """
        # Get data from query bundle
        query_dict = json.loads(query_bundle.query_str)
        original_query = query_dict['sub_question']
        tool_name = query_dict['tool_name']
        
        # Rewrite original query to n queries
        generated_queries = generate_queries(self._llm, original_query, num_queries=self._num_generated_queries)
        
        # For each generated query, retrieve relevant nodes
        retrieved_nodes = []
        for query in generated_queries:
            if len(query) == 0:
                continue
            retrieved_nodes.extend(self._retriever_mappings[tool_name].retrieve(query))
        
        # Fuse retrieved nodes using reciprocal rank
        fused_results = fuse_results(retrieved_nodes,
                                     similarity_top_k=self._similarity_top_k)
        return fused_results

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve self._similarity_top_k content nodes given query

        Args:
            query_bundle (QueryBundle): query bundle include query string
        """
        # Get data from query bundle
        query_dict = json.loads(query_bundle.query_str)
        original_query = query_dict['sub_question']
        tool_name = query_dict['tool_name']
        
        # Rewrite original query to n queries
        generated_queries = await agenerate_queries(llm=self._llm, query=original_query, num_queries=self._num_generated_queries)
        
        # For each generated query, retrieve relevant nodes
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for query in generated_queries:
                if len(query) == 0:
                    continue
                task = tg.create_task(self._retriever_mappings[tool_name].aretrieve(query))
                tasks.append(task)
        
        retrieved_nodes = [node for task in tasks for node in task.result()]
        
        # Fuse retrieved nodes using reciprocal rank
        fused_results = fuse_results(retrieved_nodes,
                                     similarity_top_k=self._similarity_top_k)
        return fused_results


@dataclass
class Response:
    response: str
    source_images: Optional[List] = None

    def __str__(self):
        return self.response

class CustomQueryEngine:
    def __init__(self,
                 retriever_tools: List[ToolMetadata],
                 fusion_retriever: BaseRetriever,
                 qa_prompt: PromptTemplate = None,
                 llm: LLM = None,
                 num_children: int = 3):
        self._qa_prompt = qa_prompt if qa_prompt else PromptTemplate(DEFAULT_FINAL_ANSWER_PROMPT_TMPL)
        self._llm = llm
        self._num_children = num_children
        self._sub_question_generator = LLMQuestionGenerator.from_defaults(llm=self._llm, 
                                                                          prompt_template_str=DEFAULT_SUB_QUESTION_PROMPT_TMPL)
        self._fusion_retriever = fusion_retriever
        self._retriever_tools = retriever_tools
        
    
    def query(self, query_str: str) -> Response:
        # Generate sub queries
        sub_queries = self._sub_question_generator.generate(tools=self._retriever_tools,
                                                            query=QueryBundle(query_str=query_str))

        # Dictionary to map response -> source_images
        response2images_mapping = defaultdict(set)
        
        # For each sub queries retrieve relevant image nodes
        # With fusion retriever, each sub query is rewritten to n queries -> retrieve relevant nodes for each generated query 
        # -> fuse all nodes retrieved from multiple generated queries using reciprocal rank -> get top k results
        for sub_query in sub_queries:
            retrieved_nodes = self._fusion_retriever.retrieve(QueryBundle(query_str=sub_query.model_dump_json()))
            # Using LLM to get the answer for sub query from retrieved nodes
            for retrieved_node in retrieved_nodes:
                response2images_mapping[str(self._llm.complete([sub_query.sub_question, Image.open(retrieved_node.node.resolve_image())]))].add(retrieved_node.node.image)
                
        # Synthesize results
        synthesized_text, source_images = synthesize_results(queries=sub_queries,
                                                             contexts=response2images_mapping,
                                                             llm=self._llm,
                                                             num_children=self._num_children)
        
        final_answer = self._llm.predict(self._qa_prompt,
                                         context_str=synthesized_text,
                                         query_str=query_str)
        
        response_template = PromptTemplate("Retrieved Information:\n"
                                           "------------------------\n"
                                           "{retrieved_information}\n"
                                           "-------------------------\n\n"
                                           "Answer:\n"
                                           "{final_answer}")
        
        return Response(response=response_template.format(retrieved_information=synthesized_text, final_answer=final_answer), source_images=source_images)
    
    async def aquery(self, query_str: str):
        sub_queries = await self._sub_question_generator.agenerate(tools=self._retriever_tools,
                                                            query=QueryBundle(query_str=query_str))
        
        retrieved_subquestion_nodes = []
        async with asyncio.TaskGroup() as tg:
            for sub_query in sub_queries:
                task = tg.create_task(self._fusion_retriever.aretrieve(QueryBundle(query_str=sub_query.model_dump_json())))
                retrieved_subquestion_nodes.append([sub_query.sub_question, task])
        
        retrieved_subquestion_nodes = [[sub_question, task.result()] for sub_question, task in retrieved_subquestion_nodes]
        
        answers = []
        # For each sub queries retrieve relevant image nodes
        # With fusion retriever, each sub query is rewritten to n queries -> retrieve relevant nodes for each generated query 
        # -> fuse all nodes retrieved from multiple generated queries using reciprocal rank -> get top k results
        async with asyncio.TaskGroup() as tg:
            for sub_question, retrieved_nodes in retrieved_subquestion_nodes:
                for retrieved_node in retrieved_nodes:
                    task = tg.create_task(self._llm.acomplete([sub_question, Image.open(retrieved_node.node.resolve_image())]))
                    answers.append([task, retrieved_node.node.image])
        
        # Dictionary to map response -> source_images
        response2images_mapping = defaultdict(set)
        
        for task, image in answers:
            response2images_mapping[str(task.result())].add(image)
        
        # Synthesize results
        synthesized_text, source_images = await asynthesize_results(queries=sub_queries,
                                                       contexts=response2images_mapping,
                                                       llm=self._llm,
                                                       num_children=self._num_children)
        
        
        final_answer = await self._llm.apredict(self._qa_prompt,
                                         context_str=synthesized_text,
                                         query_str=query_str)
        
        response_template = PromptTemplate("Retrieved Information:\n"
                                           "------------------------\n"
                                           "{retrieved_information}\n"
                                           "-------------------------\n\n"
                                           "Answer:\n"
                                           "{final_answer}")
        
        return Response(response=response_template.format(retrieved_information=synthesized_text, final_answer=final_answer), source_images=source_images)
            

        
        
        
        
        
            