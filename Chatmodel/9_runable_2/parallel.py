
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
load_dotenv()


Prompt1 = PromptTemplate(
    template= 'Generate a tweet about {topic}',
    input_variables=['topic']
)

Prompt2 = PromptTemplate(
    template= 'Generate a 200 words Linkedin post about {topic}',
    input_variables=['topic']
)
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

#  Wrap as CHAT model 
llm = ChatHuggingFace(llm=endpoint)

parser = StrOutputParser()

parallel_ch = RunnableParallel(
    {
        'tweet': RunnableSequence( Prompt1 , llm , parser),
        'linkedin' : RunnableSequence(Prompt2, llm , parser)
    }
)
res = parallel_ch.invoke({'topic':'AI'})
print(f"tweet : {res['tweet']} \n")


print(f"post : {res['linkedin']}")
