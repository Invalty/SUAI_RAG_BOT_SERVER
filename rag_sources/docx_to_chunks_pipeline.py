import json
import re
from pathlib import Path
import tiktoken
from tqdm import tqdm

INPUT_JSON = Path(r"C:\Users\gideo\project\SUAI_RAG_BOT_SERVER\parser\out_spider\spiders\parsed_docx.json")
OUTPUT_JSON = Path(r"C:\Users\gideo\project\SUAI_RAG_BOT_SERVER\rag_sources\chunks_all_docx.json")

# Создаем папку, если нужно
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# --- Функции ---
def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_by_gpt_tokens(text, chunk_size=512, overlap=50):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)

    chunks = []
    start_idx = 0
    chunk_id = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_text = enc.decode(tokens[start_idx:end_idx])

        last_space = chunk_text.rfind(" ")
        if last_space != -1 and end_idx != len(tokens):
            chunk_text_trimmed = chunk_text[:last_space]
            new_end_idx = start_idx + len(enc.encode(chunk_text_trimmed))
            chunk_text = chunk_text_trimmed
        else:
            new_end_idx = end_idx

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

        start_idx = max(new_end_idx - overlap, new_end_idx)
        chunk_id += 1

    return chunks

# --- Основной скрипт ---
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    documents_dict = json.load(f)

all_chunks = []
global_chunk_id = 0  # для уникальности chunk_id по всем документам

for doc_id, text in tqdm(documents_dict.items(), desc="Chunking documents"):
    if isinstance(text, list):
        text = " ".join(map(str, text))
    elif not isinstance(text, str):
        text = str(text)

    text = normalize_text(text)
    chunks = chunk_by_gpt_tokens(text, chunk_size=512, overlap=50)

    for chunk in chunks:
        chunk["document_id"] = doc_id
        # глобальный chunk_id
        chunk["chunk_id"] = global_chunk_id
        chunk["chunk_uid"] = f"chunk_{global_chunk_id}"
        global_chunk_id += 1
        all_chunks.append(chunk)

# Сохраняем JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"chunks": all_chunks}, f, ensure_ascii=False, indent=2)

print(f"Все документы с чанками сохранены в {OUTPUT_JSON}")
