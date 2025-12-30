from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
 
loader = PyPDFLoader("Chatmodel/10_Document_loader/dl-curriculum.pdf")
docs = loader.load()
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separator=''
)
res = splitter.split_documents(docs)
#print(res)
print(res[0].page_content)