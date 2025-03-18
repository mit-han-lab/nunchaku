import os
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import SanaTransformer2DModel
from diffusers.configuration_utils import register_to_config
from huggingface_hub import utils
from torch import nn

from .utils import NunchakuModelLoaderMixin
from ..._C import QuantizedSanaModel, utils as cutils

SVD_RANK = 32


class NunchakuSanaTransformerBlocks(nn.Module):
    def __init__(self, m: QuantizedSanaModel, dtype: torch.dtype, device: str | torch.device):
        super(NunchakuSanaTransformerBlocks, self).__init__()
        self.m = m
        self.dtype = dtype
        self.device = device

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        skip_first_layer: Optional[bool] = False
    ):

        batch_size = hidden_states.shape[0]
        img_tokens = hidden_states.shape[1]
        txt_tokens = encoder_hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        assert encoder_attention_mask is not None
        assert encoder_attention_mask.shape == (batch_size, 1, txt_tokens)

        mask = encoder_attention_mask.reshape(batch_size, txt_tokens)
        nunchaku_encoder_hidden_states = encoder_hidden_states[mask > -9000]

        cu_seqlens_txt = F.pad((mask > -9000).sum(dim=1).cumsum(dim=0), pad=(1, 0), value=0).to(torch.int32)
        cu_seqlens_img = torch.arange(
            0, (batch_size + 1) * img_tokens, img_tokens, dtype=torch.int32, device=self.device
        )

        if height is None and width is None:
            height = width = int(img_tokens**0.5)
        elif height is None:
            height = img_tokens // width
        elif width is None:
            width = img_tokens // height
        assert height * width == img_tokens

        return (
            self.m.forward(
                hidden_states.to(self.dtype).to(self.device),
                nunchaku_encoder_hidden_states.to(self.dtype).to(self.device),
                timestep.to(self.dtype).to(self.device),
                cu_seqlens_img.to(self.device),
                cu_seqlens_txt.to(self.device),
                height,
                width,
                batch_size % 3 == 0,  # pag is set when loading the model, FIXME: pag_scale == 0
                True,  # TODO: find a way to detect if we are doing CFG
                skip_first_layer,
            )
            .to(original_dtype)
            .to(original_device)
        )

    def forward_layer_at(
            self,
            idx: int,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
    ):
        batch_size = hidden_states.shape[0]
        img_tokens = hidden_states.shape[1]
        txt_tokens = encoder_hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        assert encoder_attention_mask is not None
        assert encoder_attention_mask.shape == (batch_size, 1, txt_tokens)

        mask = encoder_attention_mask.reshape(batch_size, txt_tokens)
        nunchaku_encoder_hidden_states = encoder_hidden_states[mask > -9000]

        cu_seqlens_txt = F.pad((mask > -9000).sum(dim=1).cumsum(dim=0), pad=(1, 0), value=0).to(torch.int32)
        cu_seqlens_img = torch.arange(
            0, (batch_size + 1) * img_tokens, img_tokens, dtype=torch.int32, device=self.device
        )

        if height is None and width is None:
            height = width = int(img_tokens**0.5)
        elif height is None:
            height = img_tokens // width
        elif width is None:
            width = img_tokens // height
        assert height * width == img_tokens

        return (
            self.m.forward_layer(
                idx,
                hidden_states.to(self.dtype).to(self.device),
                nunchaku_encoder_hidden_states.to(self.dtype).to(self.device),
                timestep.to(self.dtype).to(self.device),
                cu_seqlens_img.to(self.device),
                cu_seqlens_txt.to(self.device),
                height,
                width,
                batch_size % 3 == 0,  # pag is set when loading the model, FIXME: pag_scale == 0
                True,  # TODO: find a way to detect if we are doing CFG
            )
            .to(original_dtype)
            .to(original_device)
        )


class NunchakuSanaTransformer2DModel(SanaTransformer2DModel, NunchakuModelLoaderMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: Optional[int] = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
    ) -> None:
        # set num_layers to 0 to avoid creating transformer blocks
        self.original_num_layers = num_layers
        super(NunchakuSanaTransformer2DModel, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=0,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            caption_channels=caption_channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_size=sample_size,
            patch_size=patch_size,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            interpolation_scale=interpolation_scale,
        )

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        device = kwargs.get("device", "cuda")
        pag_layers = kwargs.get("pag_layers", [])
        precision = kwargs.get("precision", "int4")
        assert precision in ["int4", "fp4"]
        transformer, transformer_block_path = cls._build_model(pretrained_model_name_or_path, **kwargs)
        transformer.config["num_layers"] = transformer.original_num_layers
        m = load_quantized_module(
            transformer, transformer_block_path, device=device, pag_layers=pag_layers, use_fp4=precision == "fp4"
        )
        transformer.inject_quantized_module(m, device)
        return transformer

    def inject_quantized_module(self, m: QuantizedSanaModel, device: str | torch.device = "cuda"):
        self.transformer_blocks = torch.nn.ModuleList([NunchakuSanaTransformerBlocks(m, self.dtype, device)])
        return self


def load_quantized_module(
    net: SanaTransformer2DModel,
    path: str,
    device: str | torch.device = "cuda",
    pag_layers: int | list[int] | None = None,
    use_fp4: bool = False,
) -> QuantizedSanaModel:
    if pag_layers is None:
        pag_layers = []
    elif isinstance(pag_layers, int):
        pag_layers = [pag_layers]
    device = torch.device(device)
    assert device.type == "cuda"

    m = QuantizedSanaModel()
    cutils.disable_memory_auto_release()
    m.init(net.config, pag_layers, use_fp4, net.dtype == torch.bfloat16, 0 if device.index is None else device.index)
    m.load(path)
    return m


def inject_quantized_module(
    net: SanaTransformer2DModel, m: QuantizedSanaModel, device: torch.device
) -> SanaTransformer2DModel:
    net.transformer_blocks = torch.nn.ModuleList([NunchakuSanaTransformerBlocks(m, net.dtype, device)])
    return net
