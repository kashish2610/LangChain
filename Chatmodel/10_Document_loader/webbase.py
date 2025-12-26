from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Hugging Face endpoint
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation"
)

# Wrap as chat model
llm = ChatHuggingFace(llm=endpoint)

prompt = PromptTemplate(
    template="""
Answer the following question:
{question}

Based on the following text:
{text}
""",
    input_variables=["question", "text"]
)

parser = StrOutputParser()

url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421"
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | llm | parser

print(
    chain.invoke(
        {
            "question": "What is the product that we are talking about?",
            "text": docs[0].page_content
        }
    )
)
