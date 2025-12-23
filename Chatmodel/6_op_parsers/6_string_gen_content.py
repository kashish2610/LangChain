from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation"
)
model= ChatHuggingFace(llm=llm)

# 1st  promt --> detailed report
temp1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']

)

# 2nd prompt--> summary
temp2=PromptTemplate(
    template='Write a 5 line summary on the following text./n {text}',
    input_variables=['topic']
)
prompt1 =temp1.invoke({'topic':'black hole'})
result = model.invoke(prompt1)
prompt2 = temp2.invoke({'text':result.content})
result1 = model.invoke(prompt2)
print(result1.content)