import os

from langchain.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("./example_data/state_of_the_union.txt")

docs = loader.load()

output = docs[0].page_content[:400]
print(output)

loader = UnstructuredFileLoader(
    "./example_data/state_of_the_union.txt", mode="elements"
)

docs = loader.load()

output = docs[:5]
print(output)

loader = UnstructuredFileLoader(
    "./example_data/layout-parser-paper-fast.pdf", strategy="fast", mode="elements"
)

docs = loader.load()

output = docs[:5]
print(output)

loader = UnstructuredFileLoader(
    "./example_data/layout-parser-paper.pdf", mode="elements"
)

docs = loader.load()

output = docs[:5]
print(output)

from langchain.document_loaders import UnstructuredAPIFileLoader

filenames = ["example_data/fake.docx", "example_data/fake-email.eml"]

loader = UnstructuredAPIFileLoader(
    file_path=filenames[0],
    api_key=os.environ['UNSTRUCTURED_API_KEY'],
)

docs = loader.load()

output = docs[0]
print(output)

loader = UnstructuredAPIFileLoader(
    file_path=filenames,
    api_key=os.environ['UNSTRUCTURED_API_KEY'],
)

docs = loader.load()

output = docs[0]
print(output)
