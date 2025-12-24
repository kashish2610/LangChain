from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation"
)
pro1 =PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables= ['topic']

)
pro2 = PromptTemplate(

    template=' generate a 5 one line pointer from the following text/n {text}',
    input_variables=['text']
)
model= ChatHuggingFace(llm=llm)

parser=StrOutputParser()

chain = pro1 | model | parser | pro2 | model | parser

res=chain.invoke({'topic':'Ai'})
chain.get_graph().print_ascii()
print(res)