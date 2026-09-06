# ⚡ Vectorless RAG (ZeroVec AI)

A multimodal, vectorless Retrieval-Augmented Generation (RAG) Streamlit application. Instead of relying on dense vector embeddings and vector databases, **Vectorless RAG** builds a hierarchical document tree summary using [PageIndex](https://pageindex.ai), uses a high-capacity text LLM via [OpenRouter](https://openrouter.ai) to reason over the tree structure for precise section retrieval, and uses a Vision-Language Model (**VLM**) to answer questions directly from rendered PDF page images.

---

## 🌟 Key Features

- **Vectorless Retrieval**: Eliminates chunking and embedding issues by maintaining document hierarchy and tree summaries.
- **Dual-Model Multimodal Pipeline**:
  - **Tree Search (Text Model)**: `nvidia/nemotron-3-ultra-550b-a55b:free` navigates structured JSON document trees for top-node selection.
  - **Visual Answering (VLM)**: `minimax/minimax-m3:free` processes high-resolution PDF page images for accurate, visual-context grounded answers.
- **Interactive Streamlit Interface**: Real-time PDF document upload, chat UI, node retrieval breakdown, and rendered page image context previews.

---

## 🔑 Prerequisites & API Keys

You will need API keys for two services:

1. **PageIndex API Key**:
   - Register and obtain your key at: [https://pageindex.ai](https://pageindex.ai)
2. **OpenRouter API Key**:
   - Register and generate your API key at: [https://openrouter.ai/keys](https://openrouter.ai/keys)

---

## 🚀 Setup & Installation

### 1. Clone & Navigate to Project
```bash
cd /home/sarthak/Documents/project/vectorless-rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Open `.env` and populate your API keys:
```env
PAGEINDEX_API_KEY=your_pageindex_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## 🖥️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
vectorless-rag/
├── app.py              # Streamlit application UI & interaction flow
├── config.py           # Configuration (models, API endpoints, paths, & env loading)
├── .env                # Secret environment variables (API keys - gitignored)
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules for secrets and temporary data
├── llm_client.py       # OpenRouter & PageIndex API client integrations
├── pdf_processor.py    # PyMuPDF page rendering & PageIndex tree indexing helpers
├── prompts.py          # Prompt templates for tree search & VLM answer generation
├── requirements.txt    # Required Python package dependencies
├── README.md           # Project documentation
├── data/               # Input PDF documents directory
└── pdf_images/         # Rendered JPEG page images cache (auto-generated)
```
