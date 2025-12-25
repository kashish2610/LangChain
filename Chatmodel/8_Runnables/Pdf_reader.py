from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/sample.txt", encoding="utf-8")
documents = loader.load()

load_dotenv()

# -----------------------------
# 1. PDF LOAD
# -----------------------------
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

# -----------------------------
# 2. SPLIT DOCUMENTS
# -----------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# -----------------------------
# 3. EMBEDDINGS (HuggingFace)
# -----------------------------
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 4. VECTOR STORE
# -----------------------------
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)

# -----------------------------
# 5. RETRIEVER
# -----------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# 6. LLM (HuggingFace CHAT MODEL)
# -----------------------------
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

llm = ChatHuggingFace(llm=endpoint)

# -----------------------------
# 7. PROMPT
# -----------------------------
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# -----------------------------
# 8. PARSER
# -----------------------------
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# -----------------------------
# 9. RAG CHAIN (Sequence)
# -----------------------------
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# -----------------------------
# 10. ASK QUESTION
# -----------------------------
query = input("Ask a question from the PDF: ")
answer = rag_chain.invoke(query)

print("\nAnswer:\n", answer)
