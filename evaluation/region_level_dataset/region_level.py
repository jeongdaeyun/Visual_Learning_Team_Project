import torch
import clip
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import sys
import os
from tqdm import tqdm
sys.path.append('../../src/open_clip')

from model import CLIP 
from open_clip import create_model_and_transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dac_path = "/your_weight"

json_path = "please_download_ours_json_file"
image_folder = "please_download_ours_image"
output_folder = "./output_folder/"

model_clip, preprocess_clip = clip.load("RN50", device=device)

model_trained, preprocess_train, _ = create_model_and_transforms(
    model_name="RN50",
    pretrained="openai",
    device=device,
    precision="fp32",
    lora=4
)

checkpoint = torch.load(dac_path, map_location=device)

dac_state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))        
model_trained.load_state_dict(dac_state_dict, strict=True)
model = model_trained

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess_train(image).unsqueeze(0).to(device)
   
   
def get_text_embeddings(texts):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features

def get_image_embedding(image_tensor):
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
    return image_features

def cosine_similarity(image_features, text_features):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return torch.matmul(image_features, text_features.T)


def extract_phrases_reuon(json_path, target_image_id):
    with open(json_path, "r") as f:
        data = json.load(f)
    text_list = []
     
    for item in data:
        regions = item.get("regions", [])
        for region in regions:
            image_id = region.get("image_id")
            if image_id == target_image_id:
                phrase = region.get("phrase", "")
                if phrase:
                    text_list.append(phrase)
    return text_list 


def find_best_matching_text_and_save(tag_json_path, image_folder, output_folder):
    with open(tag_json_path, "r") as f:
        tag_data = json.load(f)

    results_attribute = []
    results_relation = []
    results_object = []

    for key, tag_info in tqdm(tag_data.items()):
        
        image_path = os.path.join(image_folder, f"{key}.jpg")

        caption = tag_info["caption"]
        negative = tag_info["negative"]
        phrase_type = tag_info["type"].strip()

        image_tensor_clip = preprocess_image(image_path)
        image_features = get_image_embedding(image_tensor_clip)
        
        text_features = get_text_embeddings([caption, negative])
        similarity_scores = cosine_similarity(image_features, text_features)
        
        print(similarity_scores)
        best_match_index = similarity_scores.argmax().item()
        predict_phrase = [caption, negative][best_match_index]
        best_score = similarity_scores[0, best_match_index].item()

        result = {
            "key": key,
            "gt_caption": caption,
            "negative_caption": negative,
            "predict_phrase": predict_phrase,
            "score": best_score
        }
        if phrase_type == "Attribute":
            results_attribute.append(result)
        elif phrase_type == "Relation":
            results_relation.append(result)
        elif phrase_type == "Object":
            results_object.append(result)

        # break
    
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, f"result_attribute.json"), "w") as f:
        json.dump(results_attribute, f, indent=4)
    with open(os.path.join(output_folder, f"result_relation.json"), "w") as f:
        json.dump(results_relation, f, indent=4)
    with open(os.path.join(output_folder, f"result_object.json"), "w") as f:
        json.dump(results_object, f, indent=4)

    print(f"Results saved to {output_folder}")


find_best_matching_text_and_save(tag_json_path, image_folder, output_folder)

print("end")