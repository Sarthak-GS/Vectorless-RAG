import time
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
import pageindex.utils as utils
from pageindex import PageIndexClient

from config import IMAGE_ROOT


def extract_pdf_page_images(pdf_path: Path, output_dir: Path) -> Tuple[Dict[int, str], int]:
    """Render each page of a PDF into JPEG image file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_document = fitz.open(str(pdf_path))

    page_images: Dict[int, str] = {}
    total_pages = len(pdf_document)

    for page_number in range(total_pages):
        page = pdf_document.load_page(page_number)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        image_path = output_dir / f"page_{page_number + 1}.jpg"
        pix.save(str(image_path))
        page_images[page_number + 1] = str(image_path)

    pdf_document.close()
    return page_images, total_pages


def get_page_images_for_nodes(
    node_list: List[str], node_map: Dict, page_images: Dict[int, str]
) -> List[str]:
    """Get paths of page images matching retrieved tree node list."""
    image_paths: List[str] = []
    seen_pages: set = set()

    for node_id in node_list:
        if node_id not in node_map:
            continue
        node_info = node_map[node_id]
        for page_num in range(node_info["start_index"], node_info["end_index"] + 1):
            if page_num in page_images and page_num not in seen_pages:
                image_paths.append(page_images[page_num])
                seen_pages.add(page_num)

    return image_paths


def prepare_document(pi_client: PageIndexClient, pdf_path: Path, session_doc_cache: Dict) -> Dict:
    """Index PDF document using PageIndex client and extract images."""
    cache_key = str(pdf_path.resolve())
    if cache_key in session_doc_cache:
        return session_doc_cache[cache_key]

    image_dir = IMAGE_ROOT / pdf_path.stem
    page_images, total_pages = extract_pdf_page_images(pdf_path, image_dir)

    doc_id = pi_client.submit_document(str(pdf_path))["doc_id"]
    for _ in range(30):
        if pi_client.is_retrieval_ready(doc_id):
            tree = pi_client.get_tree(doc_id, node_summary=True)["result"]
            node_map = utils.create_node_mapping(tree, include_page_ranges=True, max_page=total_pages)
            payload = {
                "doc_id": doc_id,
                "tree": tree,
                "page_images": page_images,
                "total_pages": total_pages,
                "node_map": node_map,
            }
            session_doc_cache[cache_key] = payload
            return payload
        time.sleep(2)

    raise TimeoutError("Document indexing is still in progress. Please try again in a few moments.")
