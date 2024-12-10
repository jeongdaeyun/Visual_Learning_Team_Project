""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import clip
import os
import warnings
from typing import Union, List
import sys
import torch
from .model import build_model_from_openai_state_dict
from .pretrained import (
    get_pretrained_url,
    list_pretrained_tag_models,
    download_pretrained_from_url,
)

__all__ = ["list_openai_models", "load_openai_model"]


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_tag_models("openai")


def load_openai_model(
    name: str,
    device: Union[str, torch.device] = "cuda:3" if torch.cuda.is_available() else "cpu",
    jit=True,
    cache_dir=None,
    lora: int = -1,
    freeze_img: bool = False,
    kqv_lora: bool = False,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights
    lora: low rank
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    ##### 이 파트
    # if name == "RN50":
    #     model_path = "/home/sliver/daeyun/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth"
    # else: 
    if get_pretrained_url(name, "openai"):
        model_path = download_pretrained_from_url(
            get_pretrained_url(name, "openai"), cache_dir=cache_dir
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {list_openai_models()}"
        )
    # print(list_openai_models())

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
        
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead"
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
    if lora > 0:
        jit = False
    
    # print(model.state_dict())
    # print(model_path)
    # sys.exit()
    
    if not jit:
        try:
            model = build_model_from_openai_state_dict(
                state_dict or model.state_dict(),
                lora=lora,
                freeze_img=freeze_img,
                kqv_lora=kqv_lora,
            ).to(device)
        except KeyError:
            
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model_from_openai_state_dict(sd).to(device)

        if str(device) == "cpu":
            model.float()                                  

        # If you want to use the RegionCLIP model's pretrained ResNet-50, then uncomment this section.
        # checkpoint_path_v = "Please Down Load Pretrain RegionCLIP Weight"
        # checkpoint = torch.load(checkpoint_path_v, map_location=device)
        # checkpoint_state_dict = checkpoint["model"]
        
        # new_state_dict = {}
        # for key, value in checkpoint_state_dict.items():
        #     if key.startswith("backbone"):
        #         new_key = key.replace("backbone", "visual")
        #     elif key.startswith("lang_encoder"):
        #         new_key = key.replace("lang_encoder.", "")
        #     else:
        #         new_key = key
        #     new_state_dict[new_key] = value

        # model.load_state_dict(new_state_dict, strict=False)
        
        return model
    
    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # ensure image_size attr available at consistent location for both jit and non-jit
    model.visual.image_size = model.input_resolution.item()
    return model
