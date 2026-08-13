# RAG Architecture Map

This diagram visualizes the Retrieval-Augmented Generation (RAG) architecture specific to the `ai-engineer` project. It maps the standard RAG concepts (Chunking, Embedding, Vector Database, Prompt Engineering) to the actual tools and functions used in the codebase.

```mermaid
graph TD
    User([Streamlit Chat Interface]) --> Query([User Query])
    
    subgraph Retrieval[Retrieval Branch]
        Chunking[Chunking<br/>Notebook 02: Filtering and selecting product metadata]
        Embedding[Embedding Model<br/>Cohere: embed-english-v3.0<br/>Function: get_embedding]
        VectorDB[(Vector Database<br/>Qdrant Collection:<br/>amazon-items-collection-00)]
        Search[Retrieval / Search<br/>Top 5 ANN Search<br/>Function: retrieve_data]
        
        Chunking -.->|Indexed before chat| Embedding
        Embedding -.->|Stored in| VectorDB
        VectorDB --> Search
    end
    
    Query -->|1. Search DB| Search
    Search --> Context([Retrieved Context<br/>5 Amazon Products])
    
    subgraph Generation[Generation Branch]
        Prompt[Prompt Engineering<br/>Function: build_prompt<br/>Injects products into strict LLM rules]
        LLM[LLM Generation<br/>Groq API: llama-3.1-8b-instant<br/>Function: generate_answer]
        
        Prompt --> LLM
    end
    
    Query -->|2. Passed to LLM| Prompt
    Context -->|3. Injected into| Prompt
    LLM --> Answer([Final Answer])
    Answer --> User
```

### How the flow works during a real chat:
1. **User Query:** A user types a message into the Streamlit interface.
2. **Retrieval:** The FastAPI backend sends that text into the **Retrieval Branch**. `get_embedding()` turns the text into numbers using Cohere, and `retrieve_data()` searches Qdrant for 5 matching products.
3. **Prompt Engineering:** Those 5 matches (`Retrieved Context`) are sent over to the **Generation Branch**, where `build_prompt()` glues them into a giant string of instructions with the user's original query.
4. **Generation:** Finally, `generate_answer()` passes that formatted string to the Groq LLM, which spits out the final conversational response that appears on the user's screen.
