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

collection_intervl2 = 'InternVL2-8B_captions'
collection_cogvlm2 = 'CogVLM2-19B_captions'


# Counting the number of entries in the collection
count_result_intervl2 = client.count(
    collection_name=collection_intervl2,
    exact=True,
)
total_points_value_intervl2 = count_result_intervl2.count
logger.info(f"Total points in the collection intervl2-8B: {total_points_value_intervl2}")

count_result_cogvlm2 = client.count(
    collection_name=collection_intervl2,
    exact=True,
)
total_points_value_cogvlm2 = count_result_cogvlm2.count
logger.info(f"Total points in the collection cogvlm2-19B: {total_points_value_cogvlm2}")

# Generate a list of IDs and select a random subset
# https://docs.python.org/3/library/random.html
if total_points_value_intervl2 == total_points_value_cogvlm2:
    logger.info("Both collections have the same number of entries.")
    ids_list = list(range(1, total_points_value_intervl2 + 1))
    seed = 42
    random.seed(seed)
    ids_subset = random.sample(ids_list, 1000)
    logger.info("Random subset:", ids_subset)
else:
    logger.error("Collections have different number of entries. Exiting...")
    exit()

# Retrieving a subset of entries from the collection
search_results_intervl2 = client.retrieve(
    collection_name=f"{collection_intervl2}",
    ids=[id for id in ids_subset],
)

search_results_cogvlm2 = client.retrieve(
    collection_name=f"{collection_cogvlm2}",
    ids=[id for id in ids_subset],
)

# print(search_results[0].payload["image_url"])
# import code; code.interact(local=dict(globals(), **locals()))
# Checking retrival results
data = []
if search_results_intervl2 and search_results_cogvlm2:
    for idx, entry_intervl2 in enumerate(search_results_intervl2):
        payload_intervl2 = entry_intervl2.payload
        image_url_intervl2 = payload_intervl2.get("image_url")

        if image_url_intervl2:
            # Load and preprocess the image
            response = requests.get(image_url_intervl2, stream=True)
            image = Image.open(response.raw).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Load and preprocess the text
            synthetic_caption_intervl2 = payload_intervl2.get("synthetic_caption", "")
        
            # Find the corresponding entry in search_results_cogvlm2
            entry_cogvlm2 = search_results_cogvlm2[idx]
            payload_cogvlm2 = entry_cogvlm2.payload
            image_url_cogvlm2 = payload_cogvlm2.get("image_url")

            assert image_url_intervl2 == image_url_cogvlm2
            # Load and preprocess the text
            synthetic_caption_cogvlm2 = payload_cogvlm2.get("synthetic_caption", "")
            # Tokenize and process text
            try:
                # https://github.com/openai/CLIP/blob/main/clip/clip.py#L205
                text_inputs = clip.tokenize([synthetic_caption_intervl2, synthetic_caption_cogvlm2], context_length=77, truncate=True).to(device)
            except RuntimeError as e:
                logger.error(f"Error tokenizing text: {e}")
                continue
            
            # Perform inference
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                
                logits_per_image, logits_per_text = model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(search_results_intervl2)} entries.")
                logger.info(f"Image URL: {image_url_intervl2}")
                logger.info(f"Label probs:, {probs}")

            data.append({
                "image_url": image_url_intervl2,
                "synthetic_caption_inter": synthetic_caption_intervl2,
                "synthetic_caption_cogvlm": synthetic_caption_cogvlm2,
                "label_probs": probs.tolist()
            })
    filename = "classification_results.json"
    filename_path = os.path.join(os.path.join(os.path.dirname(__file__), f"{filename}"))
    with open(filename_path, "w") as f:
        json.dump(data, f, indent=4)
else:
    logger.info("No entries found in the database.")

client.close()
