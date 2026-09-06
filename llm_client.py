import base64
import json
import os
import re
from typing import Dict, List, Tuple

from openai import OpenAI
from pageindex import PageIndexClient

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    PAGEINDEX_API_KEY,
    TEXT_MODEL,
    VLM_MODEL,
)


def get_clients() -> Tuple[PageIndexClient, OpenAI]:
    """Initialize and return PageIndexClient and OpenAI client for OpenRouter."""
    if not PAGEINDEX_API_KEY or not OPENROUTER_API_KEY:
        raise RuntimeError(
            "Missing API keys in .env file. "
            "Expected PAGEINDEX_API_KEY and OPENROUTER_API_KEY."
        )
    pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    or_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers=OPENROUTER_HEADERS,
    )
    return pi_client, or_client


def call_text_model(client: OpenAI, prompt: str, model: str = TEXT_MODEL) -> str:
    """Call text-only model via OpenRouter (used for tree retrieval)."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def call_vlm(
    client: OpenAI,
    prompt: str,
    image_paths: List[str] | None = None,
    model: str = VLM_MODEL,
) -> str:
    """Call vision-language model via OpenRouter using page images."""
    if image_paths:
        content: list = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            if not os.path.exists(image_path):
                continue
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def parse_tree_search_result(raw_text: str) -> Dict:
    """Parse JSON result returned from tree search prompt."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        thinking_match = re.search(
            r'"thinking"\s*:\s*"(.*?)"\s*,\s*"node_list"', text, flags=re.DOTALL
        )
        node_list_match = re.search(r'"node_list"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)

        thinking = (
            thinking_match.group(1).replace("\\n", "\n") if thinking_match else raw_text.strip()
        )
        node_list = []
        if node_list_match:
            node_list = re.findall(r'"([^"\\]+)"', node_list_match.group(1))

        return {"thinking": thinking, "node_list": node_list}
