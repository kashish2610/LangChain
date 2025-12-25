from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()
#  HF endpoint (still created)
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

#  Wrap as CHAT model 
llm = ChatHuggingFace(llm=endpoint)

prompt=PromptTemplate(
    template= 'Generate 5 intresting facts about {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()
# LLm chain
chain = RunnableSequence(
    prompt,
    llm,
    parser
)


topic= input("enter")
op = chain.invoke({'topic':topic})
print(op)