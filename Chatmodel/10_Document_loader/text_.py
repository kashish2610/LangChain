from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = TextLoader(
     "Chatmodel/10_Document_loader/cricket.txt",
    encoding="utf-8"
)
documents = loader.load()

endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

#  Wrap as CHAT model 
llm = ChatHuggingFace(llm=endpoint)

parser = StrOutputParser()
prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

chain = prompt | llm |parser
print(chain.invoke({'poem':documents[0].page_content}))


"""print(type(documents))
print(documents[0].page_content)
print(documents[0].metadata)
print(len(documents))"""

