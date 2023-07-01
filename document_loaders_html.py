from langchain.document_loaders import UnstructuredHTMLLoader

loader = UnstructuredHTMLLoader("example_data/fake-content.html")

data = loader.load()
print(data)

from langchain.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("example_data/fake-content.html")

data = loader.load()
print(data)
