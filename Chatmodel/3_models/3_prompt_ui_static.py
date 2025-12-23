from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# Load HF token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 1. Create the underlying LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",  # make sure this repo_id exists
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=512,
    temperature=0.7,
)

# 2. Wrap it in ChatHuggingFace
model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    try:
        result = model.invoke(user_input)
        st.write(result.content)
    except Exception as e:
        st.error(f"Error while calling model: {e}")
