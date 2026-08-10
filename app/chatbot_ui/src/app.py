import streamlit as st
from openai import OpenAI
from groq import Groq
from core.config import Config, config
import requests


def api_call(method, url, **kwargs):
    try:
        if method.lower() == "post":
            response = requests.post(url, **kwargs)
        else:
            response = requests.get(url, **kwargs)
        
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}
            
        if response.ok:
            return True, response_data
        
        return False, response_data
        
    except requests.exceptions.ConnectionError:
        st.error("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

def run_llm(provider, model_name, messages, max_tokens=500):

    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "Groq":
        client = Groq(api_key=config.GROQ_API_KEY)

    if provider == "Groq":
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens
        ).choices[0].message.content
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="minimal"
        ).choices[0].message.content

## Lets create a sidebar with a dropdown for the model list and providers
with st.sidebar:
    provider = st.selectbox("Select a provider", ["Groq", "OpenAI"])
    if provider == "OpenAI":
        model_name = st.selectbox("Select a model", ["gpt-5-nano", "gpt-5-mini",])
    elif provider == "Groq":
        model_name = st.selectbox("Select a model", ["llama-3.1-8b-instant"])


    # Save provider and model_name to session state
    st.session_state.provider = provider
    st.session_state.model_name = model_name
    
    st.divider()
    use_rag = st.toggle("Use Amazon RAG Search", value=True)
    st.session_state.use_rag = use_rag


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.get("use_rag"):
            # Using RAG via the backend API
            success, response_data = api_call("post", "http://api:8000/api/rag/", json={"query": prompt})
            if success:
                rag_result = response_data.get("answers", {})
                if isinstance(rag_result, dict):
                    answer = rag_result.get("answer", "Sorry, no answer could be generated.")
                else:
                    answer = rag_result
            else:
                answer = f"**Error:** {response_data.get('message', 'Unknown error')}"
        else:
            answer = run_llm(st.session_state.provider, st.session_state.model_name, st.session_state.messages)
            
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})