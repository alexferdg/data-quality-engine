from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# docker run -d -p 6333:6333 -v ./synthetic-research/qdrant-storage-research:/qdrant/storage qdrant/qdrant

client = QdrantClient(host="localhost", port=6333)

source_collection = "image_captions"
target_collection = "CogVLM2-19B_captions"

MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
embedding_model = SentenceTransformer(MODEL_NAME)

source_schema = client.get_collection(source_collection)

client.create_collection(
    collection_name=target_collection,
    vectors_config=source_schema.config.params.vectors,
)

points_generator = client.scroll(source_collection, limit=10000)

batch_size = 100

# Process points in batches
points_batch = []
for record in points_generator[0]:
    vector = embedding_model.encode(record.payload["synthetic_caption"]).tolist()
    points_batch.append(
        models.PointStruct(
            id=record.id,
            vector=vector,
            payload={
                "image_url": record.payload["image_url"],
                "image_name": record.payload["image_name"],
                "original_caption": record.payload["original_caption"],
                "synthetic_caption": record.payload["synthetic_caption"],
            }
        )
    )

    if len(points_batch) >= batch_size:
        client.upsert(
            collection_name=target_collection,
            points=points_batch
        )
        points_batch = []


if points_batch:
    client.upsert(
        collection_name=target_collection,
        points=points_batch
    )

target_points_count = client.count(collection_name=target_collection)

print(f"Number of points in the target collection: {target_points_count}")

client.close()