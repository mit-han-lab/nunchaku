import torch

from ...utils import load_state_dict_in_safetensors


def detect_format(lora: str | dict[str, torch.Tensor]) -> str:
    if isinstance(lora, str):
        tensors = load_state_dict_in_safetensors(lora, device="cpu")
    else:
        tensors = lora

    for k in tensors.keys():
        if "lora_unet_double_blocks_" in k or "lora_unet_single_blocks" in k:
            return "comfyui"
        elif ".mlp_fc" in k or "mlp_context_fc1" in k:
            return "svdquant"
        elif "double_blocks." in k or "single_blocks." in k:
            return "xlab"
        elif "transformer." in k:
            return "diffusers"
    raise ValueError("Unknown format, please provide the format explicitly.")
