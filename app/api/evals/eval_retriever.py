import os

from langsmith import Client
from qdrant_client import QdrantClient

from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from qdrant_client import QdrantClient


from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy


from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy


client = Client() 
qdrant_client = QdrantClient(url="http://localhost:6333") 


ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant"))
ragas_embeddings = LangchainEmbeddingsWrapper(CohereEmbeddings(model="embed-english-v3.0"))

async def ragas_faithfulness(run, example):
    
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        # Change this to plural 'contexts'
        retrieved_contexts=run["retrieved_contexts"] 
    )
    scorer = Faithfulness(llm=ragas_llm)
    
    return await scorer.single_turn_ascore(sample)


async def ragas_response_relevancy(run, example):
    
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        retrieved_contexts=run["retrieved_contexts"]
    )
    # Using strictness=1 to save your API rate limit!
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1)
    
    return await scorer.single_turn_ascore(sample)



async def ragas_context_precision_id_based(run, example):
    
    sample = SingleTurnSample(
        retrieved_context_ids=run["retrieved_context_ids"],
        reference_context_ids=example["reference_context_ids"]
    )
    scorer = IDBasedContextPrecision()
    
    return await scorer.single_turn_ascore(sample)