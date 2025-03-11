import os
import types

import comfy.sd
import folder_paths
import torch
from torch import nn
from transformers import T5EncoderModel

from nunchaku import NunchakuT5EncoderModel


def svdquant_t5_forward(
    self: T5EncoderModel,
    input_ids: torch.LongTensor,
    attention_mask,
    embeds=None,
    intermediate_output=None,
    final_layer_norm_intermediate=True,
    dtype: str | torch.dtype = torch.bfloat16,
    **kwargs,
):
    assert attention_mask is None
    assert intermediate_output is None
    assert final_layer_norm_intermediate
    outputs = self.encoder(input_ids=input_ids, inputs_embeds=embeds, attention_mask=attention_mask)
    hidden_states = outputs["last_hidden_state"]
    hidden_states = hidden_states.to(dtype=dtype)
    return hidden_states, None


class WrappedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input: torch.Tensor, out_dtype: torch.dtype | None = None):
        return self.embedding(input)

    @property
    def weight(self):
        return self.embedding.weight


class SVDQuantTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        prefixes = folder_paths.folder_names_and_paths["text_encoders"][0]
        local_folders = set()
        for prefix in prefixes:
            if os.path.exists(prefix) and os.path.isdir(prefix):
                local_folders_ = os.listdir(prefix)
                local_folders_ = [
                    folder
                    for folder in local_folders_
                    if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
                ]
                local_folders.update(local_folders_)
        model_paths = sorted(list(local_folders))
        return {
            "required": {
                "model_type": (["flux"],),
                "text_encoder1": (folder_paths.get_filename_list("text_encoders"),),
                "text_encoder2": (folder_paths.get_filename_list("text_encoders"),),
                "t5_min_length": (
                    "INT",
                    {"default": 512, "min": 256, "max": 1024, "step": 128, "display": "number", "lazy": True},
                ),
                "use_4bit_t5": (["disable", "enable"],),
                "int4_model": (model_paths, {"tooltip": "The name of the 4-bit T5 model."}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"

    CATEGORY = "SVDQuant"

    TITLE = "SVDQuant Text Encoder Loader"

    def load_text_encoder(
        self,
        model_type: str,
        text_encoder1: str,
        text_encoder2: str,
        t5_min_length: int,
        use_4bit_t5: str,
        int4_model: str,
    ):
        text_encoder_path1 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder1)
        text_encoder_path2 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder2)
        if model_type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        else:
            raise ValueError(f"Unknown type {model_type}")

        clip = comfy.sd.load_clip(
            ckpt_paths=[text_encoder_path1, text_encoder_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )

        if model_type == "flux":
            clip.tokenizer.t5xxl.min_length = t5_min_length

        if use_4bit_t5 == "enable":
            transformer = clip.cond_stage_model.t5xxl.transformer
            param = next(transformer.parameters())
            dtype = param.dtype
            device = param.device

            prefixes = folder_paths.folder_names_and_paths["text_encoders"][0]
            model_path = None
            for prefix in prefixes:
                if os.path.exists(os.path.join(prefix, int4_model)):
                    model_path = os.path.join(prefix, int4_model)
                    break
            if model_path is None:
                model_path = int4_model
            transformer = NunchakuT5EncoderModel.from_pretrained(model_path)
            transformer.forward = types.MethodType(svdquant_t5_forward, transformer)
            transformer.shared = WrappedEmbedding(transformer.shared)

            clip.cond_stage_model.t5xxl.transformer = (
                transformer.to(device=device, dtype=dtype) if device.type == "cuda" else transformer
            )

        return (clip,)
