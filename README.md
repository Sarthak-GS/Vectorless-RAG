# Vectorless RAG 

A multimodal, vectorless Retrieval-Augmented Generation (RAG) Streamlit application. Instead of dense vector embeddings, it builds a hierarchical document tree index via **PageIndex**, uses a text LLM to navigate the tree summary for node retrieval, and uses a Vision-Language Model (**VLM**) to reason over page images for accurate context-grounded answers.

## Project Structure

```
vectorless-rag/
├── app.py              # Main Streamlit web application
├── config.py           # Configuration (models, API endpoints, paths)
├── credentials.py      # API keys (PageIndex & OpenRouter)
├── llm_client.py       # LLM & VLM OpenRouter client integrations
├── pdf_processor.py    # PDF rendering & PageIndex document indexing helpers
├── prompts.py          # Prompt templates for tree search & VLM answer generation
├── requirements.txt    # Required Python dependencies
├── data/               # Input PDF documents directory
└── pdf_images/         # Rendered page images cache (auto-generated)
```

## Setup & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Credentials**:
   Ensure `credentials.py` contains valid API keys for `PAGEINDEX_API_KEY` and `OPENROUTER_API_KEY`.

3. **Launch Streamlit App**:
   ```bash
   streamlit run app.py
   ```
