
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
load_dotenv()


Prompt1 = PromptTemplate(
    template= 'Generate a joke about {topic}',
    input_variables=['topic']
)

Prompt2 = PromptTemplate(
    template= 'Generate a 200 words explain about {Joke}',
    input_variables=['Joke']
)
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

#  Wrap as CHAT model 
llm = ChatHuggingFace(llm=endpoint)

parser = StrOutputParser()

joke_gen = RunnableSequence(Prompt1 , llm , parser)


parallel_ch = RunnableParallel(
    {
        'Joke': RunnablePassthrough(),
        'Explain' : RunnableSequence(Prompt2, llm , parser)
    }
)

final_ch= RunnableSequence( joke_gen,parallel_ch)

res = final_ch.invoke({'topic':'AI'})
print(f"Joke : {res['Joke']} \n")


print(f"Explain : {res['Explain']}")
