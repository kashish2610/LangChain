from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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

pro1 =PromptTemplate(
    template='Generate short and simple notes from the following/n {text}',
    input_variables= ['text']

)
pro2 = PromptTemplate(

    template=' Generate a 5 short question answer from the following text \n {text}',
    input_variables=['text']
)
pro3 = PromptTemplate(
    template=' Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']

)
parser= StrOutputParser()

paralle_ch = RunnableParallel({
    'notes': pro1 | model1 | parser,
    'quiz' : pro2 | model2 | parser,

    })
merg_ch= pro3 | model1 | parser
chain = paralle_ch | merg_ch

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()