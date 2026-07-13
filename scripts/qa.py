# qa.py

from query_milvus import search  # your Milvus search function

# Example list of questions mam sent
questions_list = [
    "What is the difference between seed funding and venture capital funding?",
    "What type of company can apply?",
    "Which states has Startup policies for women",
    "What is the eligibility criteria for Skill Upgradation and Mahila Coir Yojana",
    "What is Karnataka's startup policy?",
    "What are different types of startup fundings?",
    "What is tax exemption under section 80IAC?",
    "What are different funding support for Indian startups?",
    "Can a Partnership Firm avail SISFS benefits?",
    "Can startups in Tier 2 and Tier 3 cities apply?"
]

def get_answers(questions, top_k=5):
    """
    Returns a list of dictionaries for each question with:
    - question
    - predicted_answer (top chunk text)
    - relevant_chunks (list of chunks with text, score, distance)
    """
    all_results = []

    for q in questions:
        results = search(q, top_k=top_k)
        if not results:
            all_results.append({
                "question": q,
                "predicted_answer": "No relevant context found.",
                "relevant_chunks": []
            })
            continue

        # Prepare chunks with score and distance
        chunks = []
        for r in results:
            score = r["score"]
            distance = 1 - score  # assuming Inner Product
            chunks.append({
                "text": r["text"],
                "score": score,
                "distance": distance
            })

        # Take the top chunk as predicted answer
        predicted_answer = results[0]["text"]

        all_results.append({
            "question": q,
            "predicted_answer": predicted_answer,
            "relevant_chunks": chunks
        })

    return all_results


if __name__ == "__main__":
    answers = get_answers(questions_list, top_k=5)
    for item in answers:
        print("\n=== Question ===")
        print(item["question"])
        print("\n--- Predicted Answer ---")
        print(item["predicted_answer"])
        print("\n--- Relevant Chunks ---")
        for i, c in enumerate(item["relevant_chunks"]):
            print(f"{i+1}. Score: {c['score']:.4f}, Distance: {c['distance']:.4f}")
            print(f"   Text: {c['text']}\n")
        print("="*60)
