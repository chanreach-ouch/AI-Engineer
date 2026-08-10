from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from openai import OpenAI
from groq import Groq
from api.core.config import config
from api.api.endpoints import api_router
from api.api.middleware import RequestIDMiddleware
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Engineer API", description="API for running LLMs", version="0.1.0")

app.add_middleware(RequestIDMiddleware)
app.include_router(api_router)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: List[Message]
    max_tokens: int = 500

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Engineer API! Go to /docs to see the endpoints."}

@app.post("/chat", response_model=ChatResponse)
def run_llm(request: ChatRequest):
    provider = request.provider
    model_name = request.model_name
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "Groq":
        client = Groq(api_key=config.GROQ_API_KEY)

    try:
        if provider == "Groq":
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=request.max_tokens
            ).choices[0].message.content
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=request.max_tokens,
                reasoning_effort="minimal"
            ).choices[0].message.content
        
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error calling {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
