from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = PyPDFLoader(
    "Chatmodel/10_Document_loader/dl-curriculum.pdf"
)

documents = loader.load()
print(len(documents))

