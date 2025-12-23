from langchain_huggingface import HuggingFaceEmbeddings
import sentence_transformers
print(sentence_transformers.__version__)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
document = "Delhi is the capital of India",
"Kolkata is the capital of WST Bengal",
"Paris is the capital of France"

vector = embedding.embed_documents(document)
print(str(vector)) 