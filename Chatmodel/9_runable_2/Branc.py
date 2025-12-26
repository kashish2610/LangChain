from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
load_dotenv()

def word_Counter(text):
    return len(text.split())


Prompt1 = PromptTemplate(
    template= 'Write a detail report about {topic}',
    input_variables=['topic']
)

Prompt2 = PromptTemplate(
    template= 'Summarize the following text in 100 words {text}',
    input_variables=['text']
)
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

#  Wrap as CHAT model 
llm = ChatHuggingFace(llm=endpoint)

parser = StrOutputParser()

report_gen_chain = Prompt1 | llm | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, Prompt2 | llm | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))