from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

print("Token:", os.getenv("OPENAI_API_KEY"))  # just to confirm it's loading

llm = ChatOpenAI(
    model="gpt-4o-mini",   # or any valid model you have access to
)

resp = llm.invoke("Say hello from LangChain.")
print(resp)
