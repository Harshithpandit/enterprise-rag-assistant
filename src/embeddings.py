from langchain_huggingface import HuggingFaceEmbeddings
from document_loader import chunks
from langchain_community.vectorstores import FAISS

model = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=model)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("FAISS vector store created successfully")
results = vectorstore.similarity_search(
    "How many days do I have to report a damaged product?",
    k=1
)
print("Similarity search results:")
for result in results:
    print(f"Content: {result.page_content[:200]}...")  # Print first 200 characters of the content
    print(f"Metadata: {result.metadata}")
    print("-" * 80)
