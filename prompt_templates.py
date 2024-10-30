import json
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.question_gen.prompts import build_tools_text

PREFIX = """\
Given a user question, and a list of tools, output a list of relevant sub-questions \
in json markdown that when composed can help answer the full user question:

"""

example_query_str = (
    "Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021"
)
example_tools = [
    ToolMetadata(
        name="uber_10k",
        description="Provides information about Uber financials for year 2021",
    ),
    ToolMetadata(
        name="lyft_10k",
        description="Provides information about Lyft financials for year 2021",
    ),
]
example_tools_str = build_tools_text(example_tools)
example_output = [
    SubQuestion(
        sub_question="What is the revenue growth of Uber", tool_name="uber_10k"
    ),
    SubQuestion(sub_question="What is the EBITDA of Uber", tool_name="uber_10k"),
    SubQuestion(
        sub_question="What is the revenue growth of Lyft", tool_name="lyft_10k"
    ),
    SubQuestion(sub_question="What is the EBITDA of Lyft", tool_name="lyft_10k"),
]
example_output_str = json.dumps(
    {"items": [x.model_dump() for x in example_output]}, indent=4
)

EXAMPLES = f"""\
# Example 1
<Tools>
```json
{example_tools_str}
```

<User Question>
{example_query_str}


<Output>
```json
{example_output_str}
```

"""

SUFFIX = """\
# Example 2
<Tools>
```json
{tools_str}
```

<User Question>
{query_str}

<Output>
"""

DEFAULT_SUB_QUESTION_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX

DEFAULT_GEN_PROMPT_TMPL = """\
    You are a helpful assistant that generates multiple search queries based on a \
    single input query. Generate {num_queries} search queries, one on each line, \
    related to the following input query:
    Query: {query}
    Queries:
    """
    
DEFAULT_FINAL_ANSWER_PROMPT_TMPL = """\
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    Answer: \
    """
    

SYNTHESIZE_PROMPT = """\
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the information from multiple sources and not prior knowledge,
    Summarize the information that are most relevant to the queries and return index of choices chosen to summarize.
    
    Query: {query_str}\n
    """

    
SYNTHESIZE_OUTPUT_FORMAT = """Return the output that conforms to the JSON schema below.
    Here is the output schema.
    
    {
        "properties": {
            "summarized_text": {
            "title": "Summarized Text",
            "type": "string"
            },
            "choices": {
            "items": {
                "type": "integer"
            },
            "title": "Choices",
            "type": "array"
            }
        },
        "required": [
            "summarized_text",
            "choices"
        ],
        "title": "SummarizeAnswer",
        "type": "object"
    }
    
    Answer: \
    """.replace("{", "{{").replace("}", "}}")
    
DEFAULT_SYNTHESIZE_PROMPT_TMPL = SYNTHESIZE_PROMPT + SYNTHESIZE_OUTPUT_FORMAT