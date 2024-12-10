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
mode = model_trained

# Paths
json_path = "please_download_vl_checklist_object_json_file"
image_folder = "please_download_vl_checklist_object_image"
output_folder = "./output_folder/"

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

def process_json_and_save_results(json_path, image_folder, output_folder, output_file_name):
    with open(json_path, "r") as f:
        data = json.load(f)

    results = []

    for item in tqdm(data, desc=f"Processing {os.path.basename(json_path)}"):

        image_relative_path = item[0]
        captions = item[1]
        pos_caption = captions["POS"][0]
        neg_caption = captions["NEG"][0]

        # Resolve the full image path
        image_path = os.path.join(image_folder, image_relative_path)

        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Image not found, skipping: {image_path}")
            continue

        # Process image and captions
        image_tensor = preprocess_image(image_path)
        image_features = get_image_embedding(image_tensor, mode)
        text_features = get_text_embeddings([pos_caption, neg_caption], mode)

        similarity_scores = cosine_similarity(image_features, text_features)

        best_match_index = similarity_scores.argmax().item()
        predict_caption = [pos_caption, neg_caption][best_match_index]
        best_score = similarity_scores[0, best_match_index].item()

        result = {
            "image_path": image_relative_path,
            "pos_caption": pos_caption,
            "neg_caption": neg_caption,
            "predict_caption": predict_caption,
            "score": best_score,
            "ground_truth": "POS" if predict_caption == pos_caption else "NEG"
        }
        results.append(result)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file_name)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")

json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)
    output_file_name = f"{os.path.splitext(json_file)[0]}_results.json"
    process_json_and_save_results(json_path, image_folder, output_folder, output_file_name)