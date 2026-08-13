# Project Overview

The **ai-engineer** project is an AI-powered chat application that acts as a shopping assistant. Its core feature is a Retrieval-Augmented Generation (RAG) pipeline tailored to answer user queries about Amazon items. Users can toggle between standard LLM chat (using models from Google, OpenAI, or Groq) and the specialized RAG search which grounds responses in an inventory database.

**Tech Stack & Dependencies:**
- **Language:** Python 3.12+
- **Package Manager:** `uv` (utilizing a monorepo workspace)
- **Frontend:** Streamlit (`streamlit>=1.58.0`)
- **Backend:** FastAPI (`fastapi>=0.139.0`, `uvicorn>=0.49.0`)
- **Database:** Qdrant (Vector Database via `qdrant-client>=1.18.0`)
- **LLM Integrations:** `google-genai>=2.10.0`, `openai>=2.44.0`, `groq>=1.5.0`
- **Validation & Settings:** Pydantic & Pydantic Settings
- **Observability:** LangSmith (`langsmith>=0.10.15`) for tracing LLM and retrieval calls.

---

# Folder & File Structure

The project employs a monorepo architecture utilizing `uv` workspaces, dividing the codebase into independent `api` and `chatbot_ui` modules.

```text
ai-engineer/
├── .env.example             # Template for environment variables (API keys)
├── docker-compose.yml       # Orchestrates Streamlit, FastAPI, and Qdrant services
├── Makefile                 # Utility scripts (e.g., run-docker-compose)
├── pyproject.toml           # Root workspace configuration for uv
├── uv.lock                  # Pinned dependencies for the entire workspace
├── qdrant_storage/          # Local volume for Qdrant vector database persistence
├── notebooks/               # Jupyter notebooks for data processing / prototyping
├── data/                    # Local raw/processed data storage
└── app/
    ├── api/                 # Backend FastAPI Service
    │   ├── Dockerfile
    │   ├── pyproject.toml
    │   └── src/
    │       ├── app.py       # Main FastAPI application entry point
    │       └── api/         # Core API logic
    │           ├── api/             # Routers and middleware
    │           │   ├── endpoints.py # Exposes /rag/ endpoint
    │           │   ├── middleware.py
    │           │   └── models.py    # Pydantic schemas (RAGRequest, RAGResponse)
    │           ├── agends/          # Agent and RAG logic
    │           │   └── retrieval_generation.py # Qdrant search & prompt formatting
    │           └── core/            # Configuration loaders (config.py)
    └── chatbot_ui/          # Frontend Streamlit Service
        ├── Dockerfile
        ├── pyproject.toml
        └── src/
            ├── app.py       # Main Streamlit UI and chat loop
            └── core/        # Frontend configurations
```

**Naming/Organizational Conventions:**
- The repository follows a standard Python microservice-like structure with separation of concerns. The `app/` folder splits the services, and each service has its own `src/` directory, `Dockerfile`, and `pyproject.toml`. 
- Pydantic models are centralized in `models.py`. 
- Logic is kept modular (e.g. `retrieval_generation.py` handles business logic separate from `endpoints.py`).

---

# Architecture & Data Flow

The architecture consists of three main components: a Streamlit frontend, a FastAPI backend, and a Qdrant vector database.

**Startup Lifecycle:**
1. Running `make run-docker-compose` executes `docker-compose up --build -d`.
2. Docker compose spins up three containers:
   - `qdrant`: The vector database listening on ports 6333/6334.
   - `api`: The FastAPI server built from `app/api/Dockerfile`, running on port 8000 via Uvicorn.
   - `streamlit-app`: The UI built from `app/chatbot_ui/Dockerfile`, running on port 8501.

**Data Flow (Typical User RAG Request):**
1. **User Action:** A user types a message in the Streamlit UI with the "Use Amazon RAG Search" toggle enabled.
2. **Frontend Routing:** Streamlit sends a POST request with the user's prompt to `http://api:8000/api/rag/`.
3. **Backend Processing:** FastAPI receives the request and triggers the `rag_pipeline`.
4. **Embedding:** The query is embedded using Google's `gemini-embedding-001`.
5. **Retrieval:** The embedding queries the `amazon-items-collection-00` inside Qdrant to find the top 5 most similar items.
6. **Generation:** The retrieved item IDs, descriptions, and ratings are formatted into a prompt context. This is passed to `gemini-2.5-flash` to answer the user's question based strictly on the inventory.
7. **Response Delivery:** The generated answer and the raw retrieved metadata are returned as JSON to Streamlit and displayed in the chat interface.

*Note: If the RAG toggle is disabled, Streamlit bypasses the backend API and makes a direct call to the chosen LLM provider (OpenAI, Groq, or Google) using the respective SDKs.*

---

# Key Files Deep-Dive

1. **`app/chatbot_ui/src/app.py`**
   - **Purpose:** The entire frontend interface. 
   - **Key Logic:** Contains sidebar configuration for selecting LLM providers/models, manages the chat history in `st.session_state.messages`, and handles the routing between a direct LLM call (`run_llm`) vs. calling the backend API (`api_call`).

2. **`app/api/src/app.py`**
   - **Purpose:** The entry point for the FastAPI server.
   - **Key Logic:** Initializes the FastAPI app, attaches `RequestIDMiddleware` and `CORSMiddleware`, and mounts the main API router (`api_router`) under the `/api` prefix.

3. **`app/api/src/api/api/endpoints.py`**
   - **Purpose:** Defines the backend routes.
   - **Key Logic:** Exposes the `@api_router.post("/rag/")` endpoint. It receives `RAGRequest` payloads, calls the `rag_pipeline`, and wraps the result in a `RAGResponse`.

4. **`app/api/src/api/agends/retrieval_generation.py`**
   - **Purpose:** The core business logic for the RAG pipeline.
   - **Key Functions:**
     - `get_embedding()`: Converts text to vectors using Gemini.
     - `retrieve_data()`: Queries the Qdrant database.
     - `process_context()` & `build_prompt()`: Formats results into strict LLM instructions.
     - `generate_answer()`: Triggers the final LLM response.
     - `rag_pipeline()`: Orchestrates the above functions into a single workflow.
   - **Usage:** This module is heavily decorated with `@traceable` from LangSmith to provide telemetry for each step of the RAG process.
   - **Historical Context:** The core logic inside this module was originally developed and successfully prototyped in Jupyter Notebooks before being converted and modularized into this backend API.

5. **`docker-compose.yml`**
   - **Purpose:** Local development and deployment orchestration.
   - **Key Configuration:** Wires the `streamlit-app` and `api` together, mounts local source directories for hot-reloading (`./app/api/src:/app/app/api/src`), and mounts `./qdrant_storage` to persist vector data across restarts.

---

# Configuration & Environment

- **Environment Variables:** Documented in `.env.example`.
  - API Keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`.
  - LangSmith config: `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.
  - These variables are injected into the containers via the `env_file: - .env` directive in `docker-compose.yml`.
- **Package Management:** `pyproject.toml` at the root acts as a `uv` workspace definition, orchestrating the `api` and `chatbot_ui` sub-packages. Dependencies are fully locked in `uv.lock`.
- **Scripts:** 
  - `make run-docker-compose` synchronizes dependencies (`uv sync`) and spins up the Docker containers in detached mode.
  - `make clean-notebook-outputs` strips outputs from Jupyter notebooks before committing.

---

# Dependencies Between Modules

- **Loose Coupling via HTTP:** The `chatbot_ui` depends on the `api` service strictly over HTTP (`requests.post("http://api:8000/api/rag/")`). There is no shared internal code package between the frontend and backend.
- **Data Coupling:** The `api` heavily depends on Qdrant. The `rag_pipeline` hardcodes the collection name `"amazon-items-collection-00"` and expects specific payload fields (`parent_asin`, `description`, `average_rating`). If the database schema changes, the pipeline will break.

---

# Conventions & Patterns

- **Tracing & Observability:** The backend uses LangSmith extensively. Every significant function in the `retrieval_generation.py` pipeline is decorated with `@traceable`, providing fine-grained analytics on embedding, retrieval, and prompt generation.
- **Dependency Injection / Middleware:** FastAPI utilizes custom middleware (`RequestIDMiddleware`) to attach unique IDs to incoming requests, which are then passed down into responses (`RAGResponse`).
- **Pydantic Models for I/O:** The backend strictly types its API contracts in `models.py` (e.g., `RAGRequest`, `RAGResponse`), enforcing structure on incoming and outgoing payloads.
- **Session State UI:** The Streamlit frontend heavily relies on `st.session_state` to maintain the chat loop (`messages`), toggle states (`use_rag`), and dropdown selections (`provider`, `model_name`) across re-renders.
- **Unclear / Ambiguous Code:** There is a secondary `app.py` file located at `app/api/src/api/app.py` that contains unused code for direct LLM chatting via the backend (e.g., `@app.post("/chat")`). It appears to be an older iteration or alternative entry point, as the Dockerfile points to `app/api/src/app.py` which only mounts the `/rag/` endpoint.
