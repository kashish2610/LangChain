from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# -----------------------
# 1️⃣ LOAD TRANSCRIPT
# -----------------------
video_id = "8IU7YBgpQxg"

try:
    fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in fetched_transcript.to_raw_data())
except TranscriptsDisabled:
    raise RuntimeError("No captions available.")
except Exception as e:
    raise RuntimeError(e)

# -----------------------
# 2️⃣ SPLIT TEXT
# -----------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.create_documents([transcript])

# -----------------------
# 3️⃣ EMBEDDINGS + FAISS
# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20}
)

# -----------------------
# 4️⃣ OLLAMA LLM
# -----------------------
llm = ChatOllama(
    model="llama3",
    temperature=0.2
)

# -----------------------
# 5️⃣ PROMPT
# -----------------------
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY using the transcript context.
If the answer is not explicitly mentioned, say: "Not discussed in the video."

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

# -----------------------
# 6️⃣ ASK QUESTION
# -----------------------
question = input("Ask a question: ")

docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in docs)

final_prompt = prompt.invoke({
    "context": context_text,
    "question": question
})

# -----------------------
# 7️⃣ GENERATE ANSWER
# -----------------------
answer = llm.invoke(final_prompt)
print("\nAnswer:\n", answer.content)
