from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

document_path = Path("data/documents")
pdf_files = list(document_path.glob("*.pdf"))
documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())
print(f"Loaded {len(documents)} documents from {len(list(pdf_files))} PDF files.")
print(f"First document content: {documents[0].page_content[:500]}")
print(f"First document metadata: {documents[0].metadata}")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
for i, chunk in enumerate(chunks):
    print(i, len(chunk.page_content), chunk.metadata["source"])