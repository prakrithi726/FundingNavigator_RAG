from query_milvus import search
import ollama


# Sample questions for testing
questions_list = [
    "What is the difference between seed funding and venture capital funding?",
    "What type of company can apply?",
    "Which states have Startup Policies for Women?",
    "What is the eligibility criteria for Skill Upgradation and Mahila Coir Yojana?",
    "What is Karnataka's Startup Policy?",
    "What are different types of startup funding?",
    "What is tax exemption under Section 80IAC?",
    "What are the different funding support schemes for Indian startups?",
    "Can a Partnership Firm avail SISFS benefits?",
    "Can startups in Tier 2 and Tier 3 cities apply?"
]


def generate_answer(question, context):
    """
    Generates a grounded answer using the local Llama 3 model.
    The retrieved context is treated as the primary source of truth.
    """

    prompt = f"""
You are Funding Navigator AI, an AI assistant developed to help users understand Startup India schemes, funding opportunities, tax benefits, startup policies, incubator support, and entrepreneurship initiatives.

You are given documents retrieved from a Milvus Vector Database.

Your task is to answer the user's question using ONLY the information contained in the retrieved documents.

Instructions:

• Read every retrieved source carefully before answering.

• Combine information from multiple sources whenever possible.

• Summarize the information naturally instead of copying large portions of text.

• If the retrieved documents only partially answer the question:
    - First provide all available information.
    - Then politely mention which details are not specified in the retrieved documents.

• If absolutely no relevant information exists, politely state that the retrieved documents do not contain enough information.

• Never invent facts.

• Never use outside knowledge.

• Keep the answer concise, professional and easy to understand.

• Use bullet points whenever listing schemes, policies, eligibility criteria or benefits.



Retrieved Documents:

{context}



User Question:

{question}



Helpful Answer:
"""

    response = ollama.chat(
        model="llama3-local",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


def get_answers(questions, top_k=5):
    """
    Retrieves relevant chunks from Milvus and generates answers
    using the local Llama 3 model.
    """

    all_results = []

    for question in questions:

        # Retrieve similar chunks
        results = search(question, top_k=top_k)

        if not results:
            all_results.append({
                "question": question,
                "predicted_answer": "No relevant documents were retrieved.",
                "relevant_chunks": []
            })
            continue

        retrieved_chunks = []

        context = ""

        # Build context for Llama
        for index, result in enumerate(results, start=1):

            retrieved_chunks.append({
                "text": result["text"],
                "score": result["score"],
                "distance": result["distance"]
            })

            context += (
                f"Source {index}\n"
                f"Similarity Score: {result['score']:.4f}\n\n"
                f"{result['text']}\n"
                f"\n{'-' * 70}\n\n"
            )

        # Generate answer using Llama
        generated_answer = generate_answer(question, context)

        all_results.append({
            "question": question,
            "predicted_answer": generated_answer,
            "relevant_chunks": retrieved_chunks
        })

    return all_results


if __name__ == "__main__":

    answers = get_answers(questions_list)

    for item in answers:

        print("\n" + "=" * 100)
        print("QUESTION")
        print(item["question"])

        print("\nGENERATED ANSWER")
        print(item["predicted_answer"])

        print("\nSOURCES USED")

        for index, chunk in enumerate(item["relevant_chunks"], start=1):

            print(f"\nSource {index}")
            print(f"Similarity Score : {chunk['score']:.4f}")
            print(f"Distance         : {chunk['distance']:.4f}")
            print(chunk["text"])

        print("=" * 100)
