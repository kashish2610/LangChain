from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation"
)

prompt=PromptTemplate(
    template= 'Generate 5 intresting facts about {topic}',
    input_variables=['topic']
)
model= ChatHuggingFace(llm=llm)
parser = StrOutputParser()

chain= prompt | model | parser

res= chain.invoke({'topic':'cricket'})
chain.get_graph().print_ascii()
print(res)