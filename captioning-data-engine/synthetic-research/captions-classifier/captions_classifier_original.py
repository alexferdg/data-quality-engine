import os
import logging
import torch
import clip
from PIL import Image
import random
import requests
from qdrant_client import QdrantClient
import json

# export CUDA_VISIBLE_DEVICES=3
# docker run -d -p 6333:6333 -v ./qdrant-storage:/qdrant/storage qdrant/qdrant
# https://qdrant.tech/documentation/concepts/points/

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

# Maximum token length for CLIP
MAX_TOKENS = 77

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333") 

collection_name = 'InternVL2-8B_captions'

# Counting the number of entries in the collection
count_result = client.count(
    collection_name=collection_name,
    exact=True,
)
total_points_value = count_result.count
logger.info(f"Total points in the collection: {total_points_value}")

# Generate a list of IDs and select a random subset
# https://docs.python.org/3/library/random.html
ids_list = list(range(1, total_points_value + 1))
seed = 42
random.seed(seed)
ids_subset = random.sample(ids_list, 1000)
logger.info("Random subset:", ids_subset)

# Retrieving a subset of entries from the collection
search_results = client.retrieve(
    collection_name=f"{collection_name}",
    ids=[id for id in ids_subset],
)

# print(search_results[0].payload["image_url"])
# import code; code.interact(local=dict(globals(), **locals()))
# Checking retrival results
if search_results:
    data = []
    for index, entry in enumerate(search_results):
        payload = entry.payload
        image_url = payload.get("image_url")

        if image_url:
            # Load and preprocess the image
            response = requests.get(image_url, stream=True)
            image = Image.open(response.raw).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Load and preprocess the text
            original_caption = payload.get("original_caption", "")
            synthetic_caption = payload.get("synthetic_caption", "")

            # Tokenize and process text
            try:
                # https://github.com/openai/CLIP/blob/main/clip/clip.py#L205
                text_inputs = clip.tokenize([original_caption, synthetic_caption], context_length=77, truncate=True).to(device)
            except RuntimeError as e:
                logger.error(f"Error tokenizing text: {e}")
                continue
            
            # Perform inference
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                
                logits_per_image, logits_per_text = model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            if index % 100 == 0:
                logger.info(f"Processed {index}/{len(search_results)} entries.")
                logger.info(f"Image URL: {image_url}")
                logger.info(f"Label probs:, {probs}")

            data.append({
                "image_url": image_url,
                "original_caption": original_caption,
                "synthetic_caption": synthetic_caption,
                "label_probs": probs.tolist()
            })
    filename = "classification_results_original.json"
    filename_path = os.path.join(os.path.join(os.path.dirname(__file__), f"{filename}"))
    with open(filename_path, "w") as f:
        json.dump(data, f, indent=4)
else:
    logger.info("No entries found in the database.")

client.close()
