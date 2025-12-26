
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
load_dotenv()


Prompt = PromptTemplate(
    template= 'Write a joke about {topic}',
    input_variables=['topic']
)
Prompt2 = PromptTemplate(
    template= 'eXplain the following joke {text}',
    input_variables=['text']
)
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

#  Wrap as CHAT model 
llm = ChatHuggingFace(llm=endpoint)


parser = StrOutputParser()
# LLm chain
chain = RunnableSequence(
    Prompt,
    llm,
    parser,
    Prompt2,
    llm,
    parser
)


topic= input("enter")
op = chain.invoke({'topic':topic})
print(op)