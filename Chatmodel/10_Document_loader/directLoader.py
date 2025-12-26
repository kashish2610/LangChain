from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
document=loader.load()
docs = loader.lazy_load()

for document in docs:
    print(document[0].page_content)