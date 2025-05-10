import os

import torch
from huggingface_hub import constants, hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig, T5EncoderModel

from .linear import W4Linear

class NunchakuT5EncoderModel(T5EncoderModel):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        weights_only: bool = True,
        **kwargs,
    ):

        t5_encoder = (
            super(NunchakuT5EncoderModel, cls)
            .from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                device_map="meta",
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                weights_only=weights_only,
                **kwargs,
            )
            .to(kwargs.get("torch_dtype", torch.bfloat16))
        )

        subfolder = kwargs.get("subfolder", None)
        if os.path.exists(pretrained_model_name_or_path):
            dirname = (
                pretrained_model_name_or_path
                if subfolder is None
                else os.path.join(pretrained_model_name_or_path, subfolder)
            )
            qmodel_path = os.path.join(dirname, "svdq-t5.safetensors")
        else:
            qmodel_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="svdq-t5.safetensors",
                subfolder=subfolder,
                repo_type="model",
                revision=revision,
                library_name=kwargs.get("library_name", None),
                library_version=kwargs.get("library_version", None),
                cache_dir=cache_dir,
                local_dir=kwargs.get("local_dir", None),
                user_agent=kwargs.get("user_agent", None),
                force_download=force_download,
                proxies=kwargs.get("proxies", None),
                etag_timeout=kwargs.get("etag_timeout", constants.DEFAULT_ETAG_TIMEOUT),
                token=token,
                local_files_only=local_files_only,
                headers=kwargs.get("headers", None),
                endpoint=kwargs.get("endpoint", None),
                resume_download=kwargs.get("resume_download", None),
                force_filename=kwargs.get("force_filename", None),
                local_dir_use_symlinks=kwargs.get("local_dir_use_symlinks", "auto"),
            )

        # Skip wo quantization for performance
        qlayer_suffix = tuple(kwargs.get("qlayer_suffix", (".q", ".k", ".v", ".o", ".wi_0", ".wi_1")))
        state_dict = load_file(qmodel_path)

        named_modules = {}
        for name, module in t5_encoder.named_modules():
            assert isinstance(name, str)
            if isinstance(module, nn.Linear):
                if f"{name}.qweight" in state_dict and name.endswith(qlayer_suffix):
                    print(f"Switching {name} to W4Linear")
                    qmodule = W4Linear.from_linear(module, group_size=128, init_only=True)
                    # modeling_t5.py: T5DenseGatedActDense needs dtype of weight
                    qmodule.weight = torch.empty([1], dtype=module.weight.dtype, device=module.weight.device)

                    parent_name, child_name = name.rsplit(".", 1)
                    setattr(named_modules[parent_name], child_name, qmodule)
            else:
                named_modules[name] = module

        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        t5_encoder.to_empty(device=device)
        t5_encoder.load_state_dict(state_dict, strict=False)

        return t5_encoder
