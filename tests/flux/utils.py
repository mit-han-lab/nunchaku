import os

import torch
from diffusers import FluxPipeline
from tqdm import tqdm

from ..utils import hash_str_to_int


def run_pipeline(dataset, pipeline: FluxPipeline, save_dir: str, forward_kwargs: dict = {}):
    os.makedirs(save_dir, exist_ok=True)
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)
    for row in tqdm(dataset):
        filename = row["filename"]
        prompt = row["prompt"]
        seed = hash_str_to_int(filename)
        image = pipeline(prompt, generator=torch.Generator().manual_seed(seed), **forward_kwargs).images[0]
        image.save(os.path.join(save_dir, f"{filename}.png"))
