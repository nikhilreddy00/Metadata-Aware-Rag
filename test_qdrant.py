from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_db")
res = client.scroll(collection_name="apple_sec_filings", limit=2)
for r in res[0]:
    print("Payload keys:", r.payload.keys())
    print("Payload content:", r.payload)
    print("---")
