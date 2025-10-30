import os
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTFigure
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import tiktoken
import hashlib
import multiprocessing

# Настройки
PDF_DIR = Path(r"C:\Users\gideo\project\SUAI_RAG_BOT_SERVER\rag_sources\saved_pdf\pdf")
OUTPUT_JSON = Path(r"C:\Users\gideo\project\SUAI_RAG_BOT_SERVER\rag_sources\chunks_all_pdfs.json")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


# Функции
def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def chunk_by_gpt_tokens(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    chunks, start_idx, chunk_id = [], 0, 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_text = enc.decode(tokens[start_idx:end_idx])
        last_space = chunk_text.rfind(" ")
        if last_space != -1 and end_idx != len(tokens):
            chunk_text = chunk_text[:last_space]
            end_idx = start_idx + len(enc.encode(chunk_text))
        start_offset = text.find(chunk_text)
        end_offset = start_offset + len(chunk_text)
        chunks.append({
            "chunk_uid": f"chunk_{chunk_id}",
            "chunk_id": chunk_id,
            "text": chunk_text.strip(),
            "token_count": len(enc.encode(chunk_text)),
            "start_offset": start_offset,
            "end_offset": end_offset
        })
        start_idx = max(end_idx - overlap, end_idx)
        chunk_id += 1
    return chunks


def extract_text_from_pdf(pdf_path):
    try:
        text_parts = []

        def recursive_elements(elements):
            for el in elements:
                if isinstance(el, LTTextContainer):
                    yield el
                elif isinstance(el, LTFigure):
                    yield from recursive_elements(el._objs)

        for page_layout in extract_pages(pdf_path):
            page_text = []
            for element in recursive_elements(page_layout):
                if isinstance(element, LTTextContainer):
                    page_text.append(element.get_text().strip())
            if page_text:
                text_parts.append("\n".join(page_text))

        if text_parts:
            return normalize_text("\n\n".join(text_parts))

        # fallback OCR
        ocr_texts = []
        images = convert_from_path(pdf_path)
        for img in images:
            ocr_texts.append(pytesseract.image_to_string(img, config="--psm 6").strip())
        return normalize_text("\n\n".join(ocr_texts))
    except:
        return ""


if __name__ == "__main__":
    multiprocessing.freeze_support()  # важно для Windows
    all_pdfs = list(PDF_DIR.glob("*.pdf"))
    all_chunks = []
    global_chunk_id = 0  # глобальный счетчик для уникальных chunk_id по всем документам

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_text_from_pdf, pdf): pdf for pdf in all_pdfs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Извлечение текста и чанкинг"):
            pdf_file = futures[future]
            text = future.result()
            if not text:
                continue
            chunks = chunk_by_gpt_tokens(text)
            doc_id = hashlib.sha1(str(pdf_file).encode()).hexdigest()[:10]

            for chunk in chunks:
                chunk["document_id"] = doc_id
                chunk["filename"] = pdf_file.name
                chunk["chunk_id"] = global_chunk_id
                chunk["chunk_uid"] = f"chunk_{global_chunk_id}"
                global_chunk_id += 1
                all_chunks.append(chunk)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks}, f, ensure_ascii=False, indent=2)

    print(f"Готово! Всего чанков: {len(all_chunks)}")
