import json
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# ------------------- Connect to Milvus -------------------
connections.connect("default", host="localhost", port="19530")
collection_name = "markdown_chunks"
collection = Collection(name=collection_name)

# ------------------- Embedding Model -------------------
model = SentenceTransformer("all-mpnet-base-v2")

# ------------------- Questions -------------------
questions = [
    {"Question": "What is the difference between seed funding and venture capital funding?"},
    {"Question": "What type of company can apply?"},
    {"Question": "Which states has Startup policies for women"},
    {"Question": "What is the eligibility criteria for Skill Upgradation and Mahila Coir Yojana"},
    {"Question": "What is Karnataka's startup policy?"},
    {"Question": "What are different types of startup fundings?"},
    {"Question": "What is tax exemption under section 80IAC?"},
    {"Question": "What are different funding support for Indian startups?"},
    {"Question": "Can a Partnership Firm avail SISFS benefits?"},
    {"Question": "Can startups in Tier 2 and Tier 3 cities apply?"}
]

# ------------------- Search Parameters -------------------
search_params = {"metric_type": "COSINE", "params": {"ef": 100}}

# ------------------- Perform Retrieval -------------------
results_with_chunks = []

for q in questions:
    q_text = q["Question"]
    q_emb = model.encode(q_text).tolist()

    # Search Milvus
    res = collection.search(
        data=[q_emb],
        anns_field="chunk_text_vector",
        param=search_params,
        limit=5,
        expr="",
        output_fields=["chunk_text"]  # important to get actual text
    )

    retrieved_chunks = []
    for hit in res[0]:
        retrieved_chunks.append({
            "chunk_id": hit.id,
            "chunk_text": hit.entity.get("chunk_text"),
            "score": hit.score,
            "distance": 1 - hit.score  # cosine similarity distance
        })

    results_with_chunks.append({
        "Question": q_text,
        "Retrieved_Chunks": retrieved_chunks
    })

# ------------------- Save Results -------------------
with open("qa_with_chunks.json", "w", encoding="utf-8") as f:
    json.dump(results_with_chunks, f, ensure_ascii=False, indent=4)

print("🎉 Done! Results saved to qa_with_chunks.json")
