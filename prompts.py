import json
from typing import Dict, List


def build_search_prompt(query: str, tree_without_text: List[Dict]) -> str:
    """Build prompt for text LLM to select relevant nodes from document tree summary."""
    return f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find no more than 5 tree nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""


def build_answer_prompt(query: str) -> str:
    """Build prompt for Vision-Language Model to answer based on page images."""
    return f"""
Answer the question based on the images of the document pages provided as context.

Question: {query}

Instructions:
- Provide a clear, accurate answer based ONLY on the visual context from the page images.
- Show your reasoning where helpful.
- If the answer cannot be found in the provided pages, respond with:
  "The answer is not found in the provided document."
"""
