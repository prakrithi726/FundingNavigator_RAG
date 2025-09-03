import os
import uuid
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

def recursive_chunk(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

def process_all_files(input_dir="data/cleaned", output_dir="data/chunks"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in input_dir.glob("*.md"):
        out_file = output_dir / f"{file_path.stem}_chunks.json"

        # Skip if already chunked
        if out_file.exists():
            print(f"⏩ Skipping {file_path.name}, already chunked")
            continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        text_chunks = recursive_chunk(raw_text)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_dict = {
                "chunk_id": str(uuid.uuid4()),
                "document_id": file_path.stem,
                "chunk_order": i,
                "base_url": "",
                "canonical_url": str(file_path),
                "doc_last_modified": int(file_path.stat().st_mtime),
                "content_type": "text",
                "content_source_type": "markdown",
                "language": "en",
                "chunk_text": chunk_text,
                "chunk_text_vector": [],
                "doc_version": "v1",
                "is_active": True
            }
            chunks.append(chunk_dict)

        with open(out_file, "w", encoding="utf-8") as out_f:
            json.dump(chunks, out_f, indent=2, ensure_ascii=False)

        print(f"✅ Processed {file_path.name}, {len(chunks)} chunks saved to {out_file}")

if __name__ == "__main__":
    process_all_files()
