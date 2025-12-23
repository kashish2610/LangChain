from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()
# Get your HF token from environment
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 1. Create the underlying LLM

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",  # make sure this repo_id exists
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=512,
    temperature=0.7
)

model= ChatHuggingFace(llm=llm) 
messages=[
    SystemMessage(content='You are a helpful assistance'),
    HumanMessage(content='Tell me about LangChain')
]
result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)