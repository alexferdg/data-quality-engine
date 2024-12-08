from qdrant_client import QdrantClient

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333") 

collection_name = 'image_captions'

# Counting the number of entries in the collection
count_result = client.count(
    collection_name=collection_name,
    exact=True,
)
total_points_value = count_result.count
print(f"Total points in the collection: {total_points_value}")