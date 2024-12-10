import sys
sys.path.append('../../src/open_clip')

import torch
import clip
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import os
from tqdm import tqdm

from model import CLIP 
from open_clip import create_model_and_transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_clip, preprocess_clip = clip.load("RN50", device=device)

model_trained, preprocess_train, _ = create_model_and_transforms(
    model_name="RN50",
    pretrained="openai",
    device=device,
    precision="fp32",
    lora=4
)

dac_path = "/your_weight"
checkpoint = torch.load(dac_path, map_location=device)
model_trained.load_state_dict(checkpoint["state_dict"], strict=True)

json_path = "please_download_aro_json_file"
image_folder = "please_download_aro_image"
output_folder = "./output_folder/"
mode = model_trained

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess_train(image).unsqueeze(0).to(device)
    

def get_text_embeddings(texts, model):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features

def get_image_embedding(image_tensor, model):
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
    return image_features

def cosine_similarity(image_features, text_features):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return torch.matmul(image_features, text_features.T)

def find_best_matching_text_and_save(json_path, image_folder, output_folder):
    with open(json_path, "r") as f:
        data = json.load(f)

    results = []

    for item in tqdm(data):
        image_id = item["image_id"]
        true_caption = item["true_caption"]
        false_caption = item["false_caption"]
        image_filename = item["image_path"]

        image_path = os.path.join(image_folder, image_filename)
        

        image_tensor = preprocess_image(image_path)
        image_features = get_image_embedding(image_tensor, model=mode)

        text_features = get_text_embeddings([true_caption, false_caption], model=mode)
        similarity_scores = cosine_similarity(image_features, text_features)

        best_match_index = similarity_scores.argmax().item()
        predict_caption = [true_caption, false_caption][best_match_index]
        best_score = similarity_scores[0, best_match_index].item()

        result = {
            "image_id": image_id,
            "true_caption": true_caption,
            "false_caption": false_caption,
            "predict_caption": predict_caption,
            "score": best_score,
        }
        results.append(result)

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"VG_attribute.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")


find_best_matching_text_and_save(json_path, image_folder, output_folder)

print("end")
