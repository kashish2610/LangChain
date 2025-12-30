from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
import os
load_dotenv()
# Get your HF token from environment

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# S-1a INDEXING(loader)
video_id = "8IU7YBgpQxg"

try:
    fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
    transcript_list = fetched_transcript.to_raw_data()
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
   # print(transcript)
    
except TranscriptsDisabled:
    print("No captions available for this video.")
except Exception as e:
    print(f"An error occurred: {e}")

# S-1b INDEXING(Txt splitter)
splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks= splitter.create_documents([transcript])
print(len(chunks))

# s- 1c&1d (embedding & storinging Vec Store)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",

)
vector_store= FAISS.from_documents(chunks,embeddings)
#vector_store.index_to_docstore_id

#vector_store.get_by_ids(['a0105bb4-e1fd-49e0-9246-4664795591ff'])

# 2 Retrival
retriver= vector_store.as_retriever(search_type='mmr',search_kwargs={"k":6, "fetch_k": 20})
#retriver.invoke('What is deepmind')
# S-3 Agumentation
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.2,
)

#  Wrap as CHAT model 
llm = ChatHuggingFace(llm=endpoint)
#  Wrap as CHAT model 
llm = ChatHuggingFace(llm=endpoint)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriver.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#context_text

final_prompt=prompt.invoke({"context":context_text,"question":question})

#4- Generation
answer= llm.invoke(final_prompt)
print(answer.content)

# Building a chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriver | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
main_chain.invoke('Can you summarize the video')