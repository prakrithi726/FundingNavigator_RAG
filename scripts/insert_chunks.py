import os
import json
import uuid
import warnings
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# Suppress protobuf warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# 2. Define collection schema
collection_name = "genai_chunks"

if not utility.has_collection(collection_name):
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    ]
    schema = CollectionSchema(fields, description="Chunk embeddings for GenAI project")
    collection = Collection(name=collection_name, schema=schema)
    print(f"✅ Created collection: {collection_name}")
else:
    collection = Collection(name=collection_name)
    print(f"📂 Using existing collection: {collection_name}")

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Insert chunks
chunk_dir = "data/chunks"

for file in os.listdir(chunk_dir):
    if file.endswith("_chunks.json"):
        path = os.path.join(chunk_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Handle both string-list and dict-list formats
        texts = []
        for c in chunks:
            if isinstance(c, str):  
                texts.append(c.strip())
            elif isinstance(c, dict) and "text" in c:
                texts.append(c["text"].strip())

        if not texts:
            print(f"⚠️ No valid text in {file}, skipping")
            continue

        embeddings = model.encode(texts).tolist()
        ids = [str(uuid.uuid4()) for _ in texts]

        data = [ids, embeddings, texts]
        collection.insert(data)
        print(f"✅ Inserted {len(texts)} chunks from {file}")

collection.flush()
print(" All done! Chunks inserted into Milvus.")
# 7. Create an index on the embedding field if not already created
print("\n Creating index on embedding field...")
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "IVF_FLAT",   # or IVF_SQ8, HNSW, etc.
        "metric_type": "IP",
        "params": {"nlist": 128}
    }
)

print("✅ Index created successfully!")

# 8. Load collection into memory (so it's ready for queries)
collection.load()
print("📂 Collection loaded into memory and ready for search!")
