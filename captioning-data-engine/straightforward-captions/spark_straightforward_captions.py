import re
from qdrant_client import QdrantClient, models
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, MapType

# cd captioning-data-engine/straightforward-captions
# docker run -d -p 6333:6333 -v ./qdrant-storage-stf:/qdrant/storage qdrant/qdrant

# Constants
BATCH_SIZE = 100

# Initialize Spark session
spark = (SparkSession.builder
    .appName("QdrantPayloadUpdate")
    .master("local[*]").config("spark.driver.memory", "4g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")  
    .getOrCreate())

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = 'image_captions'
collection_name_stf = 'image_captions_stf'

def remove_text(payload):
    pattern = r"^This image showcases\s*" 
    if 'synthetic_caption' in payload:
        cleaned_text = re.sub(pattern, "", payload['synthetic_caption'])
        payload['synthetic_caption'] = cleaned_text.strip()
    return payload

# Register UDF with Spark
remove_image_showcases_udf = udf(remove_text, MapType(StringType(), StringType()))

def record_to_dict(record):
    return {
        "id": record.id,
        "payload": record.payload,
        "vector": record.vector,
    }

# Function to process a batch of data
def process_batch(batch_data):

    records_as_dicts = [record_to_dict(record) for record in batch_data]
    
    df = spark.createDataFrame(records_as_dicts)
    
    df_transformed = df.withColumn("payload", remove_image_showcases_udf(col("payload")))

    df_transformed.persist()
    
    updated_payloads = df_transformed.collect()
    
    # Batch upsert to Qdrant
    qdrant_client.upsert(
        collection_name=collection_name_stf,
        points=[
            {"id": row["id"], 
             "payload": row["payload"],
             "vector": row["vector"],} 
             for row in updated_payloads
        ]
    )


def process_data_in_batches():
    offset = 0
    while True:
        # Fetch a batch of data from Qdrant
        batch_data = qdrant_client.scroll(
            collection_name=collection_name,
            limit=BATCH_SIZE,
            offset=offset,
            with_vectors=True,
        )
        
        if not batch_data[0][0]:
            break  

        process_batch(batch_data[0])
        offset += BATCH_SIZE

try:
    qdrant_client.get_collection(collection_name=collection_name_stf)
    print(f"Collection '{collection_name_stf}' already exists.")
except Exception as e:
    if "Not found: Collection" in str(e):
       
        qdrant_client.create_collection(
            collection_name=collection_name_stf,
            vectors_config=models.VectorParams(
                size=384, 
                distance=models.Distance.COSINE  
            )
        )
        print(f"Collection '{collection_name_stf}' created successfully.")
    else:
        # Raise other exceptions
        raise e
process_data_in_batches()

spark.stop()
