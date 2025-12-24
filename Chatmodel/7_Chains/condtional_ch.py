from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
llm1= HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation"
)

llm2 =HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"
)
model1= ChatHuggingFace(llm=llm1)

model2= ChatHuggingFace(llm=llm2)
parser= StrOutputParser()
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')
parser2 = PydanticOutputParser(pydantic_object=Feedback)


pro1 =PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables= ['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}

)
classifier_chain = pro1 | model1 |parser2

pro2 = PromptTemplate(

    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
pro3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', pro2 | model1 | parser),
    (lambda x:x.sentiment == 'negative', pro3 | model1 | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()