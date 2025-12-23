from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

# Load your .env file (make sure it contains HUGGINGFACEHUB_API_TOKEN)
load_dotenv("C:/Users/Kashish Sharma/Desktop/ML_pro/LangChain/Chatmodel/.env")

# Sanity check: print token
print("Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Initialize Hugging Face endpoint with a hosted chat model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",   # ✅ Hosted chat model
    task="chat-completion"
    
)

# Wrap in ChatHuggingFace for structured chat
model = ChatHuggingFace(llm=llm)

# Send a HumanMessage
result = model.invoke([HumanMessage(content="What is the smallest city of India? Answer in one line")])

# Print the model's reply
print(result.content)