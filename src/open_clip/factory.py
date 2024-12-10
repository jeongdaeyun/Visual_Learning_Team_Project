import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import torch

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import CLIP, convert_weights_to_fp16, resize_pos_embed
from .openai import load_openai_model
from .pretrained import get_pretrained_cfg, download_pretrained
from .transform import image_transform


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


def load_state_dict(checkpoint_path: str, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
    cache_dir: Optional[str] = None,
    lora: int = -1,
    freeze_img: bool = False,
    kqv_lora: bool = False,
):
    model_name = model_name.replace(
        "/", "-"
    )  # for callers using old naming with / in ViT names

    if pretrained.lower() == "openai":
        logging.info(f"Loading pretrained {model_name} from OpenAI.")
        model = load_openai_model(
            model_name,
            device=device,
            jit=jit,
            cache_dir=cache_dir,
            lora=lora,
            freeze_img=freeze_img,
            kqv_lora=kqv_lora,
        )
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        if model_name in _MODEL_CONFIGS:    #### 상단 ㅡMODEL_CONFIGS에 모델 이름 : config 파일 경로 추가 시도해보기
            logging.info(f"Loading {model_name} model config.")
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(
                f"Model config for {model_name} not found; available models {list_models()}."
            )
            raise RuntimeError(f"Model config for {model_name} not found.")

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if pretrained_image:
            if "timm_model_name" in model_cfg.get("vision_cfg", {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg["vision_cfg"]["timm_model_pretrained"] = True
            else:
                assert (
                    False
                ), "pretrained image towers currently only supported for timm models"

        model_cfg["lora"] = lora  ####  lora rank
        model = CLIP(**model_cfg) #### 모델 인스턴스 생성 -- lora 적용된 CLIP 구조 생성

        pretrained_cfg = {}    #### 사전 학습된 체크포인트 가중치 불러오기
        if pretrained:
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)   #### get_pretrianed_cfg 타고 들어가서 우리 모델 가중치랑 이름 추가해야 할 듯
            if pretrained_cfg:         #### pretrained_cfg가 위에서 존재할 경우 download_pretrained로 사전 학습된 가중치 경로 가져옴
                checkpoint_path = download_pretrained(     
                    pretrained_cfg, cache_dir=cache_dir    
                )
            elif os.path.exists(pretrained):    #### pretrained 경로에 파일이 존재할 경우, pretrained 경로에서 사전 학습된 가중치 파일을 사용할 수 있음
                checkpoint_path = pretrained    #### checkpoint_path 변수에 해당 경로를 직접 할당 

            if checkpoint_path:      #### checkpoint 경로가 설정되면, 모델에 해당 가중체 로드
                logging.info(f"Loading pretrained {model_name} weights ({pretrained}).")
                load_checkpoint(model, checkpoint_path)  # ,False    
            else:        #### pretrained_cfg가 없고 pretrained 경로에도 파일이 존재하지 않는 경우 에러 발생
                logging.warning(
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                )
                raise RuntimeError(
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                )

        model.to(device=device)
        if precision == "fp16":
            assert device.type != "cpu"
            convert_weights_to_fp16(model)

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = (
            pretrained_cfg.get("mean", None) or OPENAI_DATASET_MEAN
        )
        model.visual.image_std = pretrained_cfg.get("std", None) or OPENAI_DATASET_STD

        if jit:
            model = torch.jit.script(model)

    return model


def create_model_and_transforms(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    cache_dir: Optional[str] = None,    #### 다운로드 된 사전학습 가중치가 저장될 경로..?
    lora: int = -1,
    freeze_img: bool = False,
    kqv_lora: bool = False,
):
    ##### 모델 불러오기 
    model = create_model(   
        model_name,
        pretrained,
        precision,
        device,
        jit,
        force_quick_gelu=force_quick_gelu,
        pretrained_image=pretrained_image,
        cache_dir=cache_dir,
        lora=lora,
        freeze_img=freeze_img,
        kqv_lora=kqv_lora,
    )

    image_mean = image_mean or getattr(model.visual, "image_mean", None)
    image_std = image_std or getattr(model.visual, "image_std", None)
    # train, val에 맞게 이미지 전처리해서 가져오는 과정
    preprocess_train = image_transform(          
        model.visual.image_size, is_train=True, mean=image_mean, std=image_std
    )
    preprocess_val = image_transform(
        model.visual.image_size, is_train=False, mean=image_mean, std=image_std
    )

    return model, preprocess_train, preprocess_val


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()