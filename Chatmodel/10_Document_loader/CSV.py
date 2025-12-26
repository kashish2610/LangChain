from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="Chatmodel/10_Document_loader/Social_Network_Ads (3).csv"
)
docs = loader.load()

print(len(docs))
print(docs[1])