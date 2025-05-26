
from transformers import ViTImageProcessor, ViTModel
import numpy as np
import numpy as np
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def get_image_embedding(image_path: str) ->any:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract CLS token
    embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, 768]
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize
    print(f"Embeddings generated : {image_path}")
    return np.stack(embedding.squeeze().cpu().numpy()).astype("float32")
