from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(path="./qdrant_db")

print(f"Collection count: {client.count('apple_sec_filings')}")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="apple_sec_filings",
    embedding=embeddings,
)

print("\n--- Testing Standard Retrieval ---")
docs = vector_store.similarity_search("What are the risk factors for Apple?", k=3)
print(f"Standard retrieval found {len(docs)} docs.")
if docs:
    for d in docs:
        print(d.metadata)
        print(d.page_content[:150] + "...\n")

print("\n--- Testing Retrieval with Metadata Filter ---")
from qdrant_client.http import models
filter = models.Filter(
    must=[
        models.FieldCondition(
            key="metadata.company", # Langchain qdrant typically stores metadata under 'metadata' key in payload
            match=models.MatchValue(value="Apple Inc.")
        )
    ]
)
filtered_docs = vector_store.similarity_search("risk factors", k=2, filter=filter)
print(f"Filtered retrieval found {len(filtered_docs)} docs.")
