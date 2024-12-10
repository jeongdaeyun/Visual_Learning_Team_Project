import torch
import torch.nn as nn
import random
from spacy import load

import torch
import torch.nn.functional as F

class ClipLoss(torch.nn.Module):
    def __init__(self, logit_scale=1.0):
        super().__init__()
        self.logit_scale = logit_scale

    def forward(self, image_features, text_features):
        # 이미지와 텍스트의 유사도를 계산
        logits_per_image = self.logit_scale * torch.matmul(image_features, text_features.T)
        logits_per_text = logits_per_image.T

        # 라벨 생성
        labels = torch.arange(len(logits_per_image), device=logits_per_image.device)

        # Cross-Entropy Loss 계산
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)

        # 평균 손실 반환
        return (loss_img + loss_txt) / 2


torch.manual_seed(42)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# NLP 모델 로드 (SVO 추출용)
nlp = load("en_core_web_sm")

def extract_svo(caption):
    """Extract Subject, Verb, Object from a caption."""
    doc = nlp(caption)
    subject, verb, obj = None, None, None
    for token in doc:
        if token.dep_ == "nsubj":  
            subject = token.text
        elif token.pos_ == "VERB": 
            verb = token.text
        elif token.dep_ in ["dobj", "pobj"]:  
            obj = token.text
    return subject, verb, obj

def generate_negative_caption(svo):
    """Randomly swap two elements in SVO to create a negative example."""
    svo_list = list(svo)
    if None in svo_list:  
        return " ".join([item for item in svo_list if item])
    indices = random.sample(range(len(svo_list)), 2)  
    svo_list[indices[0]], svo_list[indices[1]] = svo_list[indices[1]], svo_list[indices[0]]
    return " ".join(svo_list)  # Negative Caption 생성

# CLIP 모델 초기화
clip_model, _, preprocess = create_model_and_transforms("ViT-B/32", pretrained="openai")
clip_model = clip_model.to(device)

# 1. Positive Caption 정의
positive_caption = "this image is a black object"

# 2. Negative Caption 생성
svo = extract_svo(positive_caption)
negative_caption = generate_negative_caption(svo)

print(f"Positive Caption: {positive_caption}")
print(f"Extracted SVO: {svo}")
print(f"Negative Caption: {negative_caption}")

# 3. 랜덤 이미지 특징 생성
batch_size = 1  
feature_dim = 512  
image_features = torch.randn(batch_size, feature_dim, device=device)

# 4. Positive Caption 및 Negative Caption 텍스트 특징 생성
# CLIP 모델의 텍스트 인코더를 사용
positive_text_features = clip_model.encode_text(positive_caption)
negative_text_features = clip_model.encode_text(negative_caption)

# 5. Logit Scale 정의
logit_scale = torch.tensor(1.0, device=device, requires_grad=True)

# 6. 손실 계산
# Positive Caption Loss
logits_pos = logit_scale * torch.matmul(image_features, positive_text_features.T)
labels = torch.arange(batch_size, device=device)
loss_pos = nn.CrossEntropyLoss()(logits_pos, labels)

# Negative Caption Loss
logits_neg = logit_scale * torch.matmul(image_features, negative_text_features.T)
negative_labels = torch.arange(batch_size, device=device)
loss_neg = nn.CrossEntropyLoss()(logits_neg, negative_labels)

# 7. Total Loss 계산
total_loss = loss_pos + loss_neg

print(f"Positive Loss: {loss_pos.item()}")
print(f"Negative Loss: {loss_neg.item()}")
print(f"Total Loss: {total_loss.item()}")
