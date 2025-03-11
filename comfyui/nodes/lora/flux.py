import logging
import os

import folder_paths
from safetensors.torch import save_file

from nunchaku.lora.flux import comfyui2diffusers, convert_to_nunchaku_flux_lowrank_dict, detect_format, xlab2diffusers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SVDQuantFluxLoraLoader")


class SVDQuantFluxLoraLoader:
    def __init__(self):
        self.cur_lora_name = "None"

    @classmethod
    def INPUT_TYPES(s):
        lora_name_list = ["None", *folder_paths.get_filename_list("loras")]

        prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
        base_model_paths = set()
        for prefix in prefixes:
            if os.path.exists(prefix) and os.path.isdir(prefix):
                base_model_paths_ = os.listdir(prefix)
                base_model_paths_ = [
                    folder
                    for folder in base_model_paths_
                    if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
                ]
                base_model_paths.update(base_model_paths_)
        base_model_paths = sorted(list(base_model_paths))

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "lora_name": (lora_name_list, {"tooltip": "The name of the LoRA."}),
                "lora_format": (
                    ["auto", "comfyui", "diffusers", "svdquant", "xlab"],
                    {"tooltip": "The format of the LoRA."},
                ),
                "base_model_name": (
                    base_model_paths,
                    {
                        "tooltip": "If the lora format is SVDQuant, this field has no use. Otherwise, the base model's state dictionary is required for converting the LoRA weights to SVDQuant."
                    },
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
                "save_converted_lora": (
                    ["disable", "enable"],
                    {
                        "tooltip": "If enabled, the converted LoRA will be saved as a .safetensors file in the save directory of your LoRA file."
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "SVDQuant FLUX.1 LoRA Loader"

    CATEGORY = "SVDQuant"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "Currently, only one LoRA nodes can be applied."
    )

    def load_lora(
        self,
        model,
        lora_name: str,
        lora_format: str,
        base_model_name: str,
        lora_strength: float,
        save_converted_lora: str,
    ):
        if self.cur_lora_name == lora_name:
            if self.cur_lora_name == "None":
                pass  # Do nothing since the lora is None
            else:
                model.model.diffusion_model.model.set_lora_strength(lora_strength)
        else:
            if lora_name == "None":
                model.model.diffusion_model.model.set_lora_strength(0)
            else:
                try:
                    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                except FileNotFoundError:
                    lora_path = lora_name
                if lora_format == "auto":
                    lora_format = detect_format(lora_path)
                if lora_format != "svdquant":
                    if lora_format == "comfyui":
                        input_lora = comfyui2diffusers(lora_path)
                    elif lora_format == "xlab":
                        input_lora = xlab2diffusers(lora_path)
                    elif lora_format == "diffusers":
                        input_lora = lora_path
                    else:
                        raise ValueError(f"Invalid LoRA format {lora_format}.")
                    prefix = os.path.join(folder_paths.models_dir, "diffusion_models")
                    base_model_path = os.path.join(prefix, base_model_name, "transformer_blocks.safetensors")
                    if not os.path.exists(base_model_path):
                        # download from huggingface
                        base_model_path = os.path.join(base_model_name, "transformer_blocks.safetensors")
                    state_dict = convert_to_nunchaku_flux_lowrank_dict(base_model_path, input_lora)

                    if save_converted_lora == "enable" and lora_format != "svdquant":
                        dirname = os.path.dirname(lora_path)
                        basename = os.path.basename(lora_path)
                        if "int4" in base_model_path:
                            precision = "int4"
                        else:
                            assert "fp4" in base_model_path
                            precision = "fp4"
                        converted_name = f"svdq-{precision}-{basename}"
                        lora_converted_path = os.path.join(dirname, converted_name)
                        if not os.path.exists(lora_converted_path):
                            save_file(state_dict, lora_converted_path)
                            logger.info(f"Saved converted LoRA to: {lora_converted_path}")
                        else:
                            logger.info(f"Converted LoRA already exists at: {lora_converted_path}")
                    model.model.diffusion_model.model.update_lora_params(state_dict)
                else:
                    model.model.diffusion_model.model.update_lora_params(lora_path)
                model.model.diffusion_model.model.set_lora_strength(lora_strength)
            self.cur_lora_name = lora_name

        return (model,)
