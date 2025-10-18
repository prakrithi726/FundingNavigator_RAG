from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Load collection
collection = Collection("genai_chunks")
collection.load()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query_text, top_k=5):
    """
    Search Milvus for the most relevant chunks based on query text.
    Returns: list of dicts with text, score (similarity), and distance.
    """
    try:
        # Create embedding for query
        query_vector = model.encode([query_text])[0].tolist()

        # Perform vector search
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},  # IP = similarity
            limit=top_k,
            output_fields=["text"],
        )

        # Convert results into a clean Python list
        hits = []
        for hit in results[0]:
            score = float(hit.score)
            distance = 1 - score  # since using Inner Product (IP), distance = 1 - similarity
            hits.append({
                "text": hit.entity.get("text", ""),
                "score": score,
                "distance": distance
            })

        return hits

    except Exception as e:
        print("❌ Search error:", e)
        return []  # Never return None
