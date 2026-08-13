from app.api.src.api.agends.retrieval_generation import rag_pipeline


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


ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant", max_retries=15))
ragas_embeddings = LangchainEmbeddingsWrapper(CohereEmbeddings(model="embed-english-v3.0"))

import asyncio

def _get_outputs(run):
    # LangSmith often wraps untraced function returns in an "output" key
    outs = run.outputs or {}
    val = outs.get("output", outs)
    return val if val is not None else {}

async def ragas_faithfulness(run, example):
    outs = _get_outputs(run)
    sample = SingleTurnSample(
        user_input=example.inputs.get("question", ""),
        response=outs.get("answer", ""),
        retrieved_contexts=outs.get("retrieved_contexts", []) 
    )
    scorer = Faithfulness(llm=ragas_llm)
    
    return await scorer.single_turn_ascore(sample)


async def ragas_response_relevancy(run, example):
    outs = _get_outputs(run)
    sample = SingleTurnSample(
        user_input=example.inputs.get("question", ""),
        response=outs.get("answer", ""),
        retrieved_contexts=outs.get("retrieved_contexts", [])
    )
    # Using strictness=1 to save your API rate limit!
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1)
    
    return await scorer.single_turn_ascore(sample)


async def ragas_context_precision_id_based(run, example):
    outs = _get_outputs(run)
    sample = SingleTurnSample(
        retrieved_context_ids=outs.get("retrieved_context_ids", []),
        reference_context_ids=example.outputs.get("reference_context_ids", [])
    )
    scorer = IDBasedContextPrecision()
    
    return await scorer.single_turn_ascore(sample)


async def ragas_context_recall_id_based(run, example):
    outs = _get_outputs(run)
    sample = SingleTurnSample(
        retrieved_context_ids=outs.get("retrieved_context_ids", []),
        reference_context_ids=example.outputs.get("reference_context_ids", [])
    )
    scorer = IDBasedContextRecall()
    
    return await scorer.single_turn_ascore(sample)

results = client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based
    ],
    experiment_prefix="retriever",
    max_concurrency=1
)