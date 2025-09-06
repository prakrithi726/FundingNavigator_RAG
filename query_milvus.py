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
    # Get query embedding
    query_vector = model.encode([query_text])[0].tolist()

    # Perform search
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )

    print(f"\n Your question: {query_text}\n")
    for i, hit in enumerate(results[0]):
        print(f"{i+1}. Score: {hit.score:.4f}")
        print(f"   Text: {hit.entity.get('text')}\n")

if __name__ == "__main__":
    print(" Milvus RAG Chatbot is ready! Type your query below (or 'exit' to quit).\n")
    while True:
        user_query = input(" Your question: ")
        if user_query.lower() in ["exit", "quit", "q"]:
            print(" Exiting chatbot. Goodbye!")
            break
        search(user_query)
