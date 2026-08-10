import cohere
from groq import Groq
from qdrant_client import QdrantClient
from langsmith import traceable , get_current_run_tree

co_client = cohere.Client()
groq_client = Groq()
@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "cohere", "ls_model_name": "embed-english-v3.0"}
)
def get_embedding(text, model="embed-english-v3.0"):
    
    response = co_client.embed(
        texts=[text],
        model=model,
        input_type="search_query"
    )
    
    return response.embeddings[0]

@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_data(query,qdrant_client, k=5):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="amazon-items-collection-00",
        query=query_embedding,
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_contexts = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_contexts.append(result.payload.get("description"))
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_contexts": retrieved_contexts,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }

@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_contexts"],
        context["retrieved_context_ratings"]
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context



@traceable(
    name="build_prompt",
    run_type="prompt"
)

def build_prompt(preprocessed_context, question):

    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products.

Context:
{preprocessed_context}

Question:
{question}
"""

    return prompt


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "groq", "ls_model_name": "llama-3.1-8b-instant"}
)
def generate_answer(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, top_k=5):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)


    final_result = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_contexts": retrieved_context["retrieved_contexts"],
        "similarity_scores": retrieved_context["similarity_scores"],
        "retrieved_context_ratings": retrieved_context["retrieved_context_ratings"],
    }
    return final_result