import os
import logging
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from threading import Lock
import time

# export CUDA_VISIBLE_DEVICES=3
# docker run -d -p 6333:6333 -v ./synthetic-research/qdrant-storage-research:/qdrant/storage qdrant/qdrant
# nohup python /home/benchmark/ax-classifier/ax-pipeline/drafts-classifier/clip/synthetic-research/captioning-qdrant/generating_qdrant.py > /home/benchmark/ax-classifier/ax-pipeline/drafts-classifier/clip/synthetic-research/captioning-qdrant/nohup_output.txt
# docker exec -it qdrant/qdrant /bin/bash
# nohup 

# Configure logging
# https://stackoverflow.com/questions/32402502/how-to-change-the-time-zone-in-python-logging  
# Configure logging to write to a file and console
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_file_path = os.path.join(os.path.dirname(__file__), f"{script_name}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DIMENSION = 384
NUM_THREADS = 60
BATCH_SIZE = 1000
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = 'InternVL2-8B_captions'
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
VLM_PIPELINE = pipeline('OpenGVLab/InternVL2-8B', backend_config=TurbomindEngineConfig(tp=1, session_len=8192))
TOTAL_SAMPLES = 10_000

# Global Variables 
COUNTER_LOCK = Lock()
counter = 0
image_counter = 0

# Directories
custom_hf_dset_save_dir = "custom_hf_dset"

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL)

# Create or reset the Qdrant collection
def initialize_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE)
    )
initialize_collection()

# Initialize embedding model
embedding_model = SentenceTransformer(MODEL_NAME)

# Functions
def generate_caption(images):
    global TOTAL_SAMPLES
    logger.info(f"Generating captions for {len(images)} images...")
    prompts = [("""Please generate a vivid and concise image caption in plain text, as if describing the scene to someone who cannot see it. 
                Start with "This image showcases..." and include the following elements: identify all objects from the most prominent to the smallest details, 
                describe their interactions or relationships (e.g., "a person standing next to a bicycle"), mention the color, size, texture, and any other relevant qualities of each object, 
                describe any actions taking place (e.g., "a dog running through a field"), provide context about the environment, such as indoors or outdoors, time of day, and weather conditions, 
                and convey the overall mood or feeling of the image, like a "peaceful sunset" or "chaotic street scene."
                """, load_image(image)) for image in images]
    

    outputs = VLM_PIPELINE(prompts)
    
    return outputs

def add_captions(image_urls, image_names, original_captions, synthetic_captions, start_index, retry_attempts=3):
    # https://qdrant.tech/documentation/concepts/points/?q=batch
    points = [
        models.PointStruct(
            id=start_index + idx,
            vector=embedding_model.encode(synthetic_caption).tolist(),
            payload={
                "image_url": image_url,
                "image_name": image_name,
                "original_caption": original_caption,
                "synthetic_caption": synthetic_caption,
            }
        )
        for idx, (image_url, image_name, original_caption, synthetic_caption) in enumerate(zip(image_urls, image_names, original_captions, synthetic_captions))
    ]

    for attempt in range(retry_attempts):
        result = client.upsert(collection_name=COLLECTION_NAME, points=points)
        if result.status == 'completed':
            logger.info("Captions upserted successfully!")
            break
        else:
            logger.error(f"Error upserting captions: {result.status}, attempt {attempt + 1} of {retry_attempts}")
            if attempt < retry_attempts - 1:
                time.sleep(2 ** attempt)
    else:
        logger.error("Failed to upsert captions after multiple attempts.")

def add_captions_batches(image_urls, image_names, original_captions, synthetic_captions, batch_size=100, num_threads=5):
    total_items = len(image_names)
    overall_start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for batch_num, start in enumerate(range(0, total_items, batch_size), start=1):
            end = min(start + batch_size, total_items)
            batch_image_urls = image_urls[start:end]
            batch_image_names = image_names[start:end]
            batch_original_captions = original_captions[start:end]
            batch_synthetic_captions = synthetic_captions[start:end]

            futures.append(executor.submit(add_captions, batch_image_urls, batch_image_names, batch_original_captions, batch_synthetic_captions, start_index=start))
            logger.info(f"Processed batch {batch_num} of {total_items // batch_size} to Qdrant DB.")

        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing a batch for Qdrant DB: {str(e)}")
        
    total_time = time.time() - overall_start_time
    logger.info(f"All batches processed for Qdrant DB in {total_time:.2f} seconds.")

def search_captions(query, top_k=5):
    query_vector = embedding_model.encode(query).tolist()
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return [(result.payload.get("synthetic_caption"), result.payload.get("image_name")) for result in search_result]

def search_captions_by_image_name(image_name):
    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="image_name",
                match=models.MatchValue(value=image_name)
            )
        ]
    )
    search_result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=search_filter,
        with_payload=True,
        limit=1
    )

    if search_result[0]:
        payload = search_result[0][0].payload
        synthetic_caption = payload.get("synthetic_caption")
        original_caption = payload.get("original_caption")
        return {"original_caption": original_caption, "synthetic_caption": synthetic_caption}
    else:
        return None

def extract_image_names(batch):
    return {"image_name": [url.split("/")[-1] for url in batch['image_url']]}


def generate_caption_batch(batch):
    captions = generate_caption(batch["image_object"])
    return {"generated_captions": [response.text for response in captions]}

# Example Usage
if __name__ == "__main__":
    dset = load_from_disk(custom_hf_dset_save_dir)

    image_urls = dset['image_url']
    image_names = dset['image_name']
    original_captions = dset["caption"]
    dset = dset.map(generate_caption_batch, batched=True, batch_size=BATCH_SIZE)
    synthetic_captions = dset["generated_captions"]
    
    add_captions_batches(image_urls, image_names, original_captions, synthetic_captions, batch_size=BATCH_SIZE, num_threads=NUM_THREADS)
    
    client.close()
