import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_ROOT = BASE_DIR / "pdf_images"

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env", override=True)

# API Keys
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Model Selection (OpenRouter)
# Step 1 – Tree-search / node retrieval: text reasoning over document tree JSON.
TEXT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Step 2 – Final answer: VLM processes document page images to generate the answer.
VLM_MODEL = "minimax/minimax-m3:free"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/vectorless-rag",
    "X-Title": "Vectorless RAG",
}
