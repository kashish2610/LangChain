from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
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

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')


prompt= template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('Summarize'):
    try:
         chain = template | model
         result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
         })
         st.write(result.content)
    except Exception as e:
         st.error(f"Error while calling model: {e}")

         