import copy
from pathlib import Path
from typing import List

import pageindex.utils as utils
import streamlit as st

from config import (
    DATA_DIR,
    IMAGE_ROOT,
    OPENROUTER_API_KEY,
    PAGEINDEX_API_KEY,
    TEXT_MODEL,
    VLM_MODEL,
)
from llm_client import (
    call_text_model,
    call_vlm,
    get_clients,
    parse_tree_search_result,
)
from pdf_processor import (
    get_page_images_for_nodes,
    prepare_document,
)
from prompts import build_answer_prompt, build_search_prompt


@st.cache_resource
def cached_get_clients():
    return get_clients()


def ensure_state() -> None:
    defaults = {
        "doc_cache": {},
        "query_count": 0,
        "messages": [],
        "selected_pdf_name": None,
        "last_uploaded_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_pdf_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted([p for p in DATA_DIR.glob("*.pdf") if p.is_file()])


def main() -> None:
    st.set_page_config(page_title="Vectorless RAG", page_icon="📄", layout="wide")
    ensure_state()

    st.markdown("<h1 style='text-align:center;'>⚡ Vectorless RAG</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:gray;'>"
        "Hierarchical tree-based document retrieval augmented with vision-language multimodal answering."
        "</p>",
        unsafe_allow_html=True,
    )

    stack = [
        ("🔎", "Nemotron Ultra 550B", "Tree Retrieval (Text)"),
        ("🖼️", "MiniMax M3", "Final Answer (VLM)"),
        ("🗂️", "PageIndex", "Doc Indexing"),
        ("🔍", "Vectorless RAG", "Retrieval"),
    ]
    cols = st.columns(len(stack))
    for col, (icon, name, role) in zip(cols, stack):
        col.markdown(
            f"<div style='text-align:center;padding:4px 3px;border:1px solid #e0e0e0;"
            f"border-radius:8px;line-height:1.15'>"
            f"<span style='font-size:1.1rem'>{icon}</span><br>"
            f"<span style='font-size:0.86rem;font-weight:600'>{name}</span><br>"
            f"<span style='font-size:0.68rem;color:gray'>{role}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    if not PAGEINDEX_API_KEY or not OPENROUTER_API_KEY:
        st.error(
            "Missing API keys. Add `PAGEINDEX_API_KEY` and `OPENROUTER_API_KEY` in `.env` file."
        )
        st.stop()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        pi_client, or_client = cached_get_clients()
    except Exception as ex:
        st.error(f"Failed to initialise clients: {ex}")
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Session")
        st.metric("Queries", st.session_state["query_count"])
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["query_count"] = 0
            st.rerun()

        st.divider()
        st.header("Document")
        uploaded_pdf = st.file_uploader(
            "Upload PDF", type=["pdf"], accept_multiple_files=False, key="pdf_uploader"
        )
        if uploaded_pdf is None:
            st.session_state["last_uploaded_signature"] = None
        else:
            upload_signature = f"{uploaded_pdf.name}:{uploaded_pdf.size}"
            if upload_signature != st.session_state.get("last_uploaded_signature"):
                file_name = Path(uploaded_pdf.name).name
                target_path = DATA_DIR / file_name

                if target_path.exists():
                    stem, suffix = target_path.stem, target_path.suffix
                    n = 1
                    while True:
                        candidate = DATA_DIR / f"{stem}_{n}{suffix}"
                        if not candidate.exists():
                            target_path = candidate
                            break
                        n += 1

                target_path.write_bytes(uploaded_pdf.getbuffer())
                st.session_state["selected_pdf_name"] = target_path.name
                st.session_state["last_uploaded_signature"] = upload_signature
                st.success(f"Uploaded {target_path.name}")

        pdf_files = get_pdf_files()
        if not pdf_files:
            st.warning("No PDFs found in data/. Upload a PDF to continue.")
            st.stop()

        default_index = 0
        selected_name = st.session_state.get("selected_pdf_name")
        if selected_name:
            for idx, pdf in enumerate(pdf_files):
                if pdf.name == selected_name:
                    default_index = idx
                    break

        selected_pdf = st.selectbox(
            "Choose a PDF", pdf_files, index=default_index, format_func=lambda p: p.name
        )
        st.session_state["selected_pdf_name"] = selected_pdf.name

        st.divider()
        st.caption(f"**Retrieval model:** `{TEXT_MODEL}`")
        st.caption(f"**VLM model:** `{VLM_MODEL}`")

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                meta = msg.get("meta", {})
                if meta.get("thinking"):
                    with st.expander("🧠 Reasoning over document tree"):
                        st.write(meta["thinking"])
                if meta.get("nodes"):
                    with st.expander("📌 Retrieved Nodes"):
                        st.dataframe(meta["nodes"], use_container_width=True)
                if meta.get("images"):
                    with st.expander("🖼️ Retrieved Page Images"):
                        st.image(
                            meta["images"],
                            caption=[Path(p).name for p in meta["images"]],
                            width=280,
                        )

    # ── Chat input ────────────────────────────────────────────────────────────
    query = st.chat_input("Ask a question about the document...")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state["messages"].append({"role": "user", "content": query})

        # Step 1 — prepare / load document index
        with st.spinner("Preparing document context…"):
            try:
                payload = prepare_document(pi_client, selected_pdf, st.session_state["doc_cache"])
            except Exception as ex:
                st.error(str(ex))
                st.stop()

        tree = payload["tree"]
        node_map = payload["node_map"]
        page_images = payload["page_images"]

        tree_without_text = utils.remove_fields(copy.deepcopy(tree), fields=["text"])
        search_prompt = build_search_prompt(query, tree_without_text)

        # Step 2 — text model reasons over the tree to select relevant nodes
        with st.spinner("🔎 Reasoning over document tree to retrieve relevant nodes…"):
            try:
                raw_search = call_text_model(or_client, search_prompt)
                search_json = parse_tree_search_result(raw_search)
                retrieved_nodes = search_json.get("node_list", [])
            except Exception as ex:
                st.error(f"Retrieval step failed: {ex}")
                st.stop()

        retrieved_images = get_page_images_for_nodes(retrieved_nodes, node_map, page_images)
        answer_prompt = build_answer_prompt(query)

        # Step 3 — VLM reads the page images and generates the final answer
        with st.spinner("🖼️ Generating final answer from PDF page images…"):
            try:
                answer = call_vlm(or_client, answer_prompt, retrieved_images)
            except Exception as ex:
                st.error(f"Answer step failed: {ex}")
                st.stop()

        rows = []
        for node_id in retrieved_nodes:
            if node_id not in node_map:
                continue
            node_info = node_map[node_id]
            node = node_info["node"]
            start_page = node_info["start_index"]
            end_page = node_info["end_index"]
            page_range = f"{start_page}" if start_page == end_page else f"{start_page}–{end_page}"
            rows.append(
                {
                    "node_id": node.get("node_id", node_id),
                    "title": node.get("title", ""),
                    "pages": page_range,
                }
            )

        meta = {
            "thinking": search_json.get("thinking", ""),
            "nodes": rows,
            "images": retrieved_images,
        }

        with st.chat_message("assistant"):
            st.write(answer)
            if meta["thinking"]:
                with st.expander("🧠 Reasoning over document tree"):
                    st.write(meta["thinking"])
            if meta["nodes"]:
                with st.expander("📌 Retrieved Nodes"):
                    st.dataframe(meta["nodes"], use_container_width=True)
            if meta["images"]:
                with st.expander("🖼️ Retrieved Page Images"):
                    st.image(
                        meta["images"],
                        caption=[Path(p).name for p in meta["images"]],
                        width=280,
                    )

        st.session_state["messages"].append({"role": "assistant", "content": answer, "meta": meta})
        st.session_state["query_count"] += 1


if __name__ == "__main__":
    main()
