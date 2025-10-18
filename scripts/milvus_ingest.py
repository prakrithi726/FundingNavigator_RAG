import os
import json
import logging
import uuid
import time
from sentence_transformers import SentenceTransformer
from milvus_storage import MilvusClient  # ✅ use MilvusClient, not MilvusStorage

logger = logging.getLogger("milvus_ingest")
logging.basicConfig(level=logging.INFO)

# Load SentenceTransformer model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_chunks(folder_path):
    """Load all JSON chunks from folder"""
    chunks = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        chunks.extend(data)
                    else:
                        logger.warning(f"Skipping {file_name}: empty or invalid")
            except Exception as e:
                logger.error(f"Failed to load {file_name}: {str(e)}")
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def prepare_chunks(chunks):
    """Add required fields & embeddings for Milvus schema"""
    prepared = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text") or chunk.get("chunk_text", "").strip()
        if not text:
            logger.warning(f"Skipping chunk {i}: empty text")
            continue

        embedding = model.encode(text).tolist()

        prepared.append({
            "chunk_id": str(uuid.uuid4()),
            "document_id": chunk.get("document_id", str(uuid.uuid4())),
            "chunk_order": chunk.get("chunk_order", i),
            "base_url": chunk.get("base_url", ""),
            "canonical_url": chunk.get("canonical_url", ""),
            "crawl_date": chunk.get("crawl_date", int(time.time())),
            "doc_last_modified": chunk.get("doc_last_modified", int(time.time())),
            "content_type": chunk.get("content_type", "text"),
            "content_source_type": chunk.get("content_source_type", "webpage"),
            "language": chunk.get("language", "en"),  # ✅ default language
            "chunk_text": text,
            "chunk_text_vector": embedding,
            "doc_version": chunk.get("doc_version", "v1"),
            "is_active": chunk.get("is_active", True)
        })
    return prepared


def main():
    # Step 1: Initialize Milvus client
    milvus_client = MilvusClient(logger=logger, collection_name="funds_collection")

    # Step 2: Load raw chunks
    chunks = load_chunks("data/chunks")

    # Step 3: Prepare chunks with defaults + embeddings
    prepared_chunks = prepare_chunks(chunks)

    # Step 4: Insert into Milvus
    res = milvus_client.insert_nodes(logger=logger, nodes=prepared_chunks)
    logger.info(f"Ingestion complete. Inserted {len(prepared_chunks)} chunks into Milvus.")


if __name__ == "__main__":
    main()
