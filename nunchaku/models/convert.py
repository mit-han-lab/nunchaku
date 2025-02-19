# -*- coding: utf-8 -*-
"""QServe state dict converter module."""

import argparse
import os

import safetensors.torch
import torch
import tqdm


def ceil_divide(x: int, divisor: int) -> int:
    """Ceiling division.

    Args:
        x (`int`):
            dividend.
        divisor (`int`):
            divisor.

    Returns:
        `int`:
            ceiling division result.
    """
    return (x + divisor - 1) // divisor


def ceil_num_groups(in_features: int, group_size: int, weight_bits: int = 4) -> int:
    """Calculate the ceiling number of quantization groups.

    Args:
        in_features (`int`):
            input channel size.
        group_size (`int`):
            quantization group size.
        weight_bits (`int`, *optional*, defaults to `4`):
            quantized weight bits.

    Returns:
        `int`:
            ceiling number of quantization groups.
    """
    assert in_features % group_size == 0, "input channel size should be divisible by group size."
    num_groups = in_features // group_size
    assert weight_bits in (4, 2, 1), "weight bits should be 4, 2, or 1."
    pack_size = 32 // weight_bits  # one INT32 contains `pack_size` elements of weights
    num_packs = ceil_divide(num_groups, pack_size)
    if group_size >= 128:
        num_packs_factor = 1
    elif group_size == 64:
        num_packs_factor = 2
    elif group_size == 32:
        num_packs_factor = 4
    else:
        raise NotImplementedError
    # make sure num_packs is a multiple of num_packs_factor
    num_packs = ceil_divide(num_packs, num_packs_factor) * num_packs_factor
    num_groups = num_packs * pack_size
    return num_groups


def pack_w4(weight: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.int32, f"quantized weight should be torch.int32, but got {weight.dtype}."
    oc, ic = weight.shape
    assert ic % 32 == 0, "input channel size should be divisible by 32."
    # [0, 1, ..., 31] -> [0, 8, 16, 24, 1, 9, 17, 25, ..., 7, 15, 23, 31]
    weight = weight.view(-1, 4, 8)
    weight = weight[:, 0] | (weight[:, 1] << 4) | (weight[:, 2] << 8) | (weight[:, 3] << 12)
    weight = weight.view(oc // 4, 4, ic // 64, 16).permute(0, 2, 1, 3).reshape(oc // 4, ic)
    return weight.to(torch.int16)


def convert_to_tinychat_w4x16y16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int = -1,
    zero_pre_scaled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a weight tensor to TinyChat W4-X16-Y16 linear weight format.

    Args:
        weight (`torch.Tensor`):
            weight tensor to be converted.
        scale (`torch.Tensor`):
            scale tensor for the weight tensor.
        zero (`torch.Tensor`):
            zero point tensor for the weight tensor.
        group_size (`int`, *optional*, defaults to `-1`):
            quantization group size.
        zero_pre_scaled (`bool`, *optional*, defaults to `False`):
            whether zero point tensor is pre-scaled.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`:
            packed quantized weight tensor, scale tensor, and zero point tensor.
    """
    dtype, device = weight.dtype, weight.device
    assert dtype in (torch.float16, torch.bfloat16), "currently tinychat only supports fp16 and bf16."
    assert scale is not None, "scale tensor is required for quantization."
    assert zero is not None, "zero point tensor is required for quantization."
    weight = weight.to(dtype=torch.float32)
    scale = scale.to(dtype=torch.float32, device=device)
    zero = zero.to(dtype=torch.float32, device=device)
    if zero_pre_scaled:
        zero = zero * scale
    oc, ic = weight.shape
    group_size = ic if group_size <= 0 else group_size
    assert group_size <= ic, "group size should be less than or equal to input channel size."
    assert ic % group_size == 0, "input channel size should be divisible by group size."
    ng = ic // group_size
    if scale.numel() == 1:
        scale = scale.view(1, 1).expand(oc, ng)
    scale = scale.reshape(oc, ng).contiguous().view(oc, ng, 1)
    if zero.numel() == 1:
        zero = zero.view(1, 1).expand(oc, ng)
    zero = zero.reshape(oc, ng).contiguous().view(oc, ng, 1)
    weight = weight.view(oc, ng, -1).add_(zero).div_(scale).round_().view(oc, ic)
    _weight = pack_w4(weight.to(torch.int32))
    _ng = ceil_num_groups(ic, group_size, weight_bits=4)
    _scale = torch.zeros((_ng, oc), dtype=dtype, device=device)
    _zero = torch.zeros((_ng, oc), dtype=dtype, device=device)
    _scale[:ng] = scale.view(oc, ng).t().to(dtype=dtype)
    _zero[:ng] = zero.view(oc, ng).t().to(dtype=dtype).neg_()
    return _weight, _scale, _zero


def convert_to_tinychat_w4x16y16_linear_state_dict(
    param_name: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int = -1,
    zero_pre_scaled: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a weight tensor to TinyChat W4-X16-Y16 linear state dictionary.

    Args:
        param_name (`str`):
            parameter name.
        weight (`torch.Tensor`):
            weight tensor to be converted.
        scale (`torch.Tensor`):
            scale tensor for the weight tensor.
        zero (`torch.Tensor`):
            zero point tensor for the weight tensor.
        group_size (`int`, *optional*, defaults to `-1`):
            quantization group size.
        zero_pre_scaled (`bool`, *optional*, defaults to `False`):
            whether zero point tensor is pre-scaled.

    Returns:
        `dict[str, torch.Tensor]`:
            state dictionary for the quantized weight tensor.
    """
    module_name = param_name[:-7]
    weight, scale, zero = convert_to_tinychat_w4x16y16_linear_weight(
        weight, scale=scale, zero=zero, group_size=group_size, zero_pre_scaled=zero_pre_scaled
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict[f"{module_name}.qweight"] = weight.cpu()
    state_dict[f"{module_name}.scales"] = scale.cpu()
    state_dict[f"{module_name}.scaled_zeros"] = zero.cpu()
    return state_dict


def convert_to_tinychat_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    group_size: int = -1,
) -> dict[str, torch.Tensor]:
    scales: dict[str, dict[tuple[int, ...], torch.Tensor]] = {}
    zeros: dict[str, tuple[torch.Tensor | None, bool]] = {}
    print("Loading scale tensors...")
    for name, tensor in tqdm.tqdm(scale_dict.items(), desc="Loading scale tensors", leave=False, dynamic_ncols=True):
        print(f"  - Loading tensor {name} (dtype: {tensor.dtype}, shape: {tensor.shape}, device: {tensor.device})")
        if name.endswith("zero"):
            # this is a zero point tensor
            zero = None if tensor is None or all(t.item() == 0 for t in tensor.flatten()) else tensor
            if name.endswith(".scaled_zero"):
                zeros[name[:-12]] = (zero, False)  # zero point tensor is post-scaled
            else:
                zeros[name[:-5]] = (zero, True)  # zero point tensor is pre-scaled
        else:
            assert ".weight.scale" in name
            # this is a scale tensor
            idx = name.index(".weight.scale")
            param_name = name[: idx + 7]
            scale_level = tuple(map(int, name[idx + 14 :].split(".")))
            scales.setdefault(param_name, {})[scale_level] = tensor
    for param_name in zeros.keys():
        assert param_name in state_dict, f"zero point tensor {param_name} not found in state dict."
        assert param_name in scales, f"scale tensor {param_name} not found in scale dict."
    converted: dict[str, torch.Tensor] = {}
    print("Converting state dict...")
    for param_name, param in tqdm.tqdm(state_dict.items(), desc="Converting state dict", dynamic_ncols=True):
        if param_name in scales:
            print(f"  - Converting {param_name} (dtype: {param.dtype}, shape: {param.shape}, device: {param.device})")
            weight = param.data.clone()
            if param_name in zeros:
                zero, zero_pre_scaled = zeros[param_name]
                zero = zero.clone() if zero is not None else None
            else:
                zero, zero_pre_scaled = None, False
            level_scales = sorted(scales[param_name].items(), key=lambda x: x[0])
            assert len(level_scales) == 1, "more than one scale levels are not supported."
            scale = level_scales[0][1].clone()
            converted.update(
                convert_to_tinychat_w4x16y16_linear_state_dict(
                    param_name,
                    weight,
                    scale=scale,
                    zero=zero,
                    group_size=group_size,
                    zero_pre_scaled=zero_pre_scaled,
                )
            )
        else:
            if isinstance(param, torch.Tensor):
                print(f"  - Copying {param_name} (dtype: {param.dtype}, shape: {param.shape}, device: {param.device})")
                converted[param_name] = param.clone().cpu()
            else:
                print(f"  - Copying {param_name} (type: {type(param)}, value: {param})")
                converted[param_name] = param
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="path to the quantization checkpoint directory.")
    parser.add_argument("--group-size", type=int, default=-1, help="quantization group size.")
    parser.add_argument("--output-root", type=str, default="", help="root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=None, help="model name.")
    parser.add_argument("--model-path", type=str, default=None, help="path to the huggingface model directory.")
    parser.add_argument("--copy-on-save", action="store_true", help="copy files on save.")
    args = parser.parse_args()
    if not args.output_root:
        args.output_root = args.quant_path
    if args.model_name is None:
        assert args.model_path is not None, "model name or path is required."
        model_name = args.model_path.rstrip(os.sep).split(os.sep)[-1]
        print(f"Model name not provided. Using model name {model_name}.")
    else:
        model_name = args.model_name
    state_dict = torch.load(
        os.path.join(args.quant_path, "model.pt"),
        map_location="cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu",
    )
    scale_dict = torch.load(os.path.join(args.quant_path, "scale.pt"), map_location="cpu")
    converted = convert_to_tinychat_state_dict(state_dict, scale_dict, group_size=args.group_size)
    model_name = f"{args.model_name}-w4a16"
    model_name += f"-g{args.group_size}" if args.group_size > 0 else "-gchn"
    output_dirpath = os.path.join(args.output_root, model_name)

    os.makedirs(output_dirpath, exist_ok=True)
    if args.model_path and os.path.exists(args.model_path):
        output_path = os.path.join(output_dirpath, "model.safetensors")
        safetensors.torch.save_file(converted, output_path)
        print(f"Quantized model checkpoint saved to {output_path}.")
        for filename in os.listdir(args.model_path):
            if filename == "tokenizer.model" or (
                filename.endswith(".json") and filename != "pytorch_model.bin.index.json"
            ):
                filepath = os.path.abspath(os.path.join(args.model_path, filename))
                if args.copy_on_save:
                    os.system(f"cp {filepath} {output_dirpath}/")
                else:
                    os.system(f"ln -s {filepath} {output_dirpath}/{filename}")
    else:
        output_path = os.path.join(output_dirpath, "tinychat-v2.pt")
        torch.save(converted, output_path)
        print(f"Quantized model checkpoint saved to {output_path}.")
    print(f"Quantized model saved to {output_dirpath}.")
