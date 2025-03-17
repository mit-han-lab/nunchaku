<div align="center" id="nunchaku_logo">
  <img src="assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>Paper</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>Website</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>Blog</b></a> | <a href="https://svdquant.mit.edu"><b>Demo</b></a> | <a href="https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c"><b>HuggingFace</b></a> | <a href="https://modelscope.cn/collections/svdquant-468e8f780c2641"><b>ModelScope</b></a> | <a href="https://github.com/mit-han-lab/ComfyUI-nunchaku"><b>ComfyUI</b></a>
</h3>

**Nunchaku** is a high-performance inference engine optimized for 4-bit diffusion models, as introduced in our paper [SVDQuant](http://arxiv.org/abs/2411.05007). For the underlying quantization library, check out [DeepCompressor](https://github.com/mit-han-lab/deepcompressor).

Join our user groups on [**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q) and [**WeChat**](./assets/wechat.jpg) to engage in discussions with the community! More details can be found [here](https://github.com/mit-han-lab/nunchaku/issues/149). If you have any questions, run into issues, or are interested in contributing, don’t hesitate to reach out!

## News

- **[2025-03-17]** 🚀 Released NVFP4 4-bit [Shuttle-Jaguar](https://huggingface.co/mit-han-lab/svdq-int4-shuttle-jaguar) and FLUX.1-tools and also upgraded the INT4 FLUX.1-tool models. Download and update your models from our [HuggingFace](https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c) or [ModelScope](https://modelscope.cn/collections/svdquant-468e8f780c2641) collections!
- **[2025-03-13]** 📦 Separate the ComfyUI node into a [standalone repository](https://github.com/mit-han-lab/ComfyUI-nunchaku) for easier installation and release node v0.1.6! Plus, [4-bit Shuttle-Jaguar](https://huggingface.co/mit-han-lab/svdq-int4-shuttle-jaguar) is now fully supported!
- **[2025-03-07]** 🚀 **Nunchaku v0.1.4 Released!** We've supported [4-bit text encoder and per-layer CPU offloading](#Low-Memory-Inference), reducing FLUX's minimum memory requirement to just **4 GiB** while maintaining a **2–3× speedup**. This update also fixes various issues related to resolution, LoRA, pin memory, and runtime stability. Check out the release notes for full details!
- **[2025-02-20]** 🚀 We release the [pre-built wheels](https://huggingface.co/mit-han-lab/nunchaku) to simplify installation! Check [here](#Installation) for the guidance!
- **[2025-02-20]** 🚀 **Support NVFP4 precision on NVIDIA RTX 5090!** NVFP4 delivers superior image quality compared to INT4, offering **~3× speedup** on the RTX 5090 over BF16. Learn more in our [blog](https://hanlab.mit.edu/blog/svdquant-nvfp4), checkout  [`examples`](./examples) for usage and try [our demo](https://svdquant.mit.edu/flux1-schnell/) online!
- **[2025-02-18]** 🔥 [**Customized LoRA conversion**](#Customized-LoRA) and [**model quantization**](#Customized-Model-Quantization) instructions are now available! **[ComfyUI](./comfyui)** workflows now support **customized LoRA**, along with **FLUX.1-Tools**!
- **[2025-02-14]** 🔥 **[LoRA conversion script](nunchaku/convert_lora.py)** is now available! [ComfyUI FLUX.1-tools workflows](./comfyui) is released!
- **[2025-02-11]** 🎉 **[SVDQuant](http://arxiv.org/abs/2411.05007) has been selected as a ICLR 2025 Spotlight! FLUX.1-tools Gradio demos are now available!** Check [here](#gradio-demos) for the usage details! Our new [depth-to-image demo](https://svdquant.mit.edu/flux1-depth-dev/) is also online—try it out!


<details>
<summary>More</summary>
  
- **[2025-02-04]** **🚀 4-bit [FLUX.1-tools](https://blackforestlabs.ai/flux-1-tools/) is here!** Enjoy a **2-3× speedup** over the original models. Check out the [examples](./examples) for usage. **ComfyUI integration is coming soon!**
- **[2025-01-23]** 🚀 **4-bit [SANA](https://nvlabs.github.io/Sana/) support is here!** Experience a 2-3× speedup compared to the 16-bit model. Check out the [usage example](./examples/int4-sana_1600m_pag.py) and the [deployment guide](app/sana/t2i) for more details. Explore our live demo at [svdquant.mit.edu](https://svdquant.mit.edu)!
- **[2025-01-22]** 🎉 [**SVDQuant**](http://arxiv.org/abs/2411.05007) has been accepted to **ICLR 2025**!
- **[2024-12-08]** Support [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Please check [mit-han-lab/ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) for the usage.
- **[2024-11-07]** 🔥 Our latest **W4A4** Diffusion model quantization work [**SVDQuant**](https://hanlab.mit.edu/projects/svdquant) is publicly released! Check [**DeepCompressor**](https://github.com/mit-han-lab/deepcompressor) for the quantization library.

</details>

## Overview

![teaser](./assets/teaser.jpg)
SVDQuant is a post-training quantization technique for 4-bit weights and activations that well maintains visual fidelity. On 12B FLUX.1-dev, it achieves 3.6× memory reduction compared to the BF16 model. By eliminating CPU offloading, it offers 8.7× speedup over the 16-bit model when on a 16GB laptop 4090 GPU, 3× faster than the NF4 W4A16 baseline. On PixArt-∑, it demonstrates significantly superior visual quality over other W4A4 or even W4A8 baselines. "E2E" means the end-to-end latency including the text encoder and VAE decoder.

**SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models**<br>
[Muyang Li](https://lmxyy.me)\*, [Yujun Lin](https://yujunlin.com)\*, [Zhekai Zhang](https://hanlab.mit.edu/team/zhekai-zhang)\*, [Tianle Cai](https://www.tianle.website/#/), [Xiuyu Li](https://xiuyuli.com), [Junxian Guo](https://github.com/JerryGJX), [Enze Xie](https://xieenze.github.io), [Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), and [Song Han](https://hanlab.mit.edu/songhan) <br>
*MIT, NVIDIA, CMU, Princeton, UC Berkeley, SJTU, and Pika Labs* <br>

<p align="center">
  <img src="assets/demo.gif" width="100%"/>
</p>

## Method

#### Quantization Method -- SVDQuant

![intuition](./assets/intuition.gif)Overview of SVDQuant. Stage1: Originally, both the activation $\boldsymbol{X}$ and weights $\boldsymbol{W}$ contain outliers, making 4-bit quantization challenging.  Stage 2: We migrate the outliers from activations to weights, resulting in the updated activation $\hat{\boldsymbol{X}}$ and weights $\hat{\boldsymbol{W}}$. While $\hat{\boldsymbol{X}}$ becomes easier to quantize, $\hat{\boldsymbol{W}}$ now becomes more difficult. Stage 3: SVDQuant further decomposes $\hat{\boldsymbol{W}}$ into a low-rank component $\boldsymbol{L}_1\boldsymbol{L}_2$ and a residual $\hat{\boldsymbol{W}}-\boldsymbol{L}_1\boldsymbol{L}_2$ with SVD. Thus, the quantization difficulty is alleviated by the low-rank branch, which runs at 16-bit precision. 

#### Nunchaku Engine Design

![engine](./assets/engine.jpg) (a) Naïvely running low-rank branch with rank 32 will introduce 57% latency overhead due to extra read of 16-bit inputs in *Down Projection* and extra write of 16-bit outputs in *Up Projection*. Nunchaku optimizes this overhead with kernel fusion. (b) *Down Projection* and *Quantize* kernels use the same input, while *Up Projection* and *4-Bit Compute* kernels share the same output. To reduce data movement overhead, we fuse the first two and the latter two kernels together.


## Performance

![efficiency](./assets/efficiency.jpg)SVDQuant reduces the model size of the 12B FLUX.1 by 3.6×. Additionally, Nunchaku, further cuts memory usage of the 16-bit model by 3.5× and delivers 3.0× speedups over the NF4 W4A16 baseline on both the desktop and laptop NVIDIA RTX 4090 GPUs. Remarkably, on laptop 4090, it achieves in total 10.1× speedup by eliminating CPU offloading.

## Installation

### Wheels

**Note:** For native Windows users, we have released a preliminary wheel to ease the installation. See [here](https://github.com/mit-han-lab/nunchaku/issues/169) for more details!

#### For Windows WSL Users

To install and use WSL (Windows Subsystem for Linux), follow the instructions [here](https://learn.microsoft.com/en-us/windows/wsl/install). You can also install WSL directly by running the following commands in PowerShell:
```shell
wsl --install # install the latest WSL
wsl # launch WSL
```

#### Prerequisites for all users
Before installation, ensure you have [PyTorch>=2.5](https://pytorch.org/) installed. For example, you can use the following command to install PyTorch 2.6:

```shell
pip install torch==2.6 torchvision==0.21 torchaudio==2.6
```

#### Installing nunchaku
Once PyTorch is installed, you can directly install `nunchaku` from our [Hugging Face repository](https://huggingface.co/mit-han-lab/nunchaku/tree/main). Be sure to select the appropriate wheel for your Python and PyTorch version. For example, for Python 3.11 and PyTorch 2.6:

```shell
pip install https://huggingface.co/mit-han-lab/nunchaku/resolve/main/nunchaku-0.1.4+torch2.6-cp311-cp311-linux_x86_64.whl
```

**Note**: NVFP4 wheels are not currently available because PyTorch has not officially supported CUDA 11.8. To use NVFP4, you will need **Blackwell GPUs (e.g., 50-series GPUs)** and must **build from source**.

### Build from Source

**Note**:

*  Ensure your CUDA version is **≥ 12.2 on Linux** and **≥ 12.6 on Windows**.

*  For Windows users, please refer to [this issue](https://github.com/mit-han-lab/nunchaku/issues/6) for the instruction. Please upgrade your MSVC compiler to the latest version.

*  We currently support only NVIDIA GPUs with architectures sm_86 (Ampere: RTX 3090, A6000), sm_89 (Ada: RTX 4090), and sm_80 (A100). See [this issue](https://github.com/mit-han-lab/nunchaku/issues/1) for more details.


1. Install dependencies:
  ```shell
  conda create -n nunchaku python=3.11
  conda activate nunchaku
  pip install torch torchvision torchaudio
  pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub
  pip install peft opencv-python gradio spaces GPUtil  # For gradio demos
  ```

 To enable NVFP4 on Blackwell GPUs (e.g., 50-series GPUs), please install nightly PyTorch with CUDA 12.8. The installation command can be:

  ```shell
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  ```

2. Install `nunchaku` package:
    Make sure you have `gcc/g++>=11`. If you don't, you can install it via Conda:

    ```shell
    conda install -c conda-forge gxx=11 gcc=11
    ```

    Then build the package from source:
    ```shell
    git clone https://github.com/mit-han-lab/nunchaku.git
    cd nunchaku
    git submodule init
    git submodule update
    pip install -e . --no-build-isolation
    ```

**[Optional]** You can verify your installation by running: `python -m nunchaku.test`. This command will download and run our 4-bit FLUX.1-schnell model.

## Usage Example

In [examples](examples), we provide minimal scripts for running INT4 [FLUX.1](https://github.com/black-forest-labs/flux) and [SANA](https://github.com/NVlabs/Sana) models with Nunchaku. For example, the [script](examples/int4-flux.1-dev.py) for [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) is as follows:

```python
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipeline("A cat holding a sign that says hello world", num_inference_steps=50, guidance_scale=3.5).images[0]
image.save("flux.1-dev.png")
```

Specifically, `nunchaku` shares the same APIs as [diffusers](https://github.com/huggingface/diffusers) and can be used in a similar way.

### Low Memory Inference

To further reduce GPU memory usage, you can use our 4-bit T5 encoder along with CPU offloading, requiring a minimum of just 4GiB of memory. The usage is also simple in the diffusers' way. For example, the [script](examples/int4-flux.1-dev-qencoder.py) for FLUX.1-dev is as follows:

```python
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    "mit-han-lab/svdq-int4-flux.1-dev", offload=True
)  # set offload to False if you want to disable offloading
text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", text_encoder_2=text_encoder_2, transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
pipeline.enable_sequential_cpu_offload()  # remove this line if you want to disable the CPU offloading
image = pipeline("A cat holding a sign that says hello world", num_inference_steps=50, guidance_scale=3.5).images[0]
image.save("flux.1-dev.png")
```

## Customized LoRA

![lora](./assets/lora.jpg)

[SVDQuant](http://arxiv.org/abs/2411.05007) seamlessly integrates with off-the-shelf LoRAs without requiring requantization. To convert your LoRA safetensors to our format, use the following command:

```shell
python -m nunchaku.lora.flux.convert \
  --quant-path mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors \
  --lora-path aleksa-codes/flux-ghibsky-illustration/lora.safetensors \
  --output-root ./nunchaku_loras \
  --lora-name svdq-int4-flux.1-dev-ghibsky
```

Argument Details:

- `--quant-path`: The path to the quantized base model. It can be a local path or a remote Hugging Face model. For example, you can use [`mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors`](https://huggingface.co/mit-han-lab/svdq-int4-flux.1-dev/blob/main/transformer_blocks.safetensors) for FLUX.1-dev.

- `--lora-path`: The path to your LoRA safetensors, which can also be a local or remote Hugging Face model.

- `--lora-format`: Specifies the LoRA format. Supported formats include:
  - `auto`: The default option. Automatically detects the appropriate LoRA format.
  - `diffusers` (e.g., [aleksa-codes/flux-ghibsky-illustration](https://huggingface.co/aleksa-codes/flux-ghibsky-illustration))
  - `comfyui` (e.g., [Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch))
  - `xlab` (e.g., [XLabs-AI/flux-RealismLora](https://huggingface.co/XLabs-AI/flux-RealismLora))
  
- `--output-root`: Specifies the output directory for the converted LoRA.

- `--lora-name`: Sets the name of the converted LoRA file (without `.safetensors` extension).

After converting your LoRA, you can use your converted weight with:

```python
transformer.update_lora_params(path_to_your_converted_lora)
transformer.set_lora_strength(lora_strength)
```

`path_to_your_lora` can also be a remote HuggingFace path. In [examples/int4-flux.1-dev-lora.py](examples/int4-flux.1-dev-lora.py), we provide a minimal example script for running [Ghibsky](https://huggingface.co/aleksa-codes/flux-ghibsky-illustration) LoRA with SVDQuant's INT4 FLUX.1-dev:

```python
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

### LoRA Related Code ###
transformer.update_lora_params(
    "mit-han-lab/svdquant-lora-collection/svdq-int4-flux.1-dev-ghibsky.safetensors"
)  # Path to your converted LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(1)  # Your LoRA strength here
### End of LoRA Related Code ###

image = pipeline(
    "GHIBSKY style, cozy mountain cabin covered in snow, with smoke curling from the chimney and a warm, inviting light spilling through the windows",
    num_inference_steps=25,
    guidance_scale=3.5,
).images[0]
image.save("flux.1-dev-ghibsky.png")
```

**For ComfyUI users, we have implemented a node to convert the LoRA weights on the fly. All you need to do is specify the correct LoRA format. Please refer to [mit-han-lab/ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) for more details.**

## ComfyUI

Please refer to [mit-han-lab/ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) for the usage in [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Gradio Demos

* FLUX.1 Models
  * Text-to-image: see [`app/flux.1/t2i`](app/flux.1/t2i).
  * Sketch-to-Image ([pix2pix-Turbo](https://github.com/GaParmar/img2img-turbo)): see [`app/flux.1/sketch`](app/flux.1/sketch).
  * Depth/Canny-to-Image ([FLUX.1-tools](https://blackforestlabs.ai/flux-1-tools/)): see [`app/flux.1/depth_canny`](app/flux.1/depth_canny).
  * Inpainting ([FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev)): see [`app/flux.1/fill`](app/flux.1/fill).
  * Redux ([FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev)): see [`app/flux.1/redux`](app/flux.1/redux).
* SANA:
  * Text-to-image: see [`app/sana/t2i`](app/sana/t2i).

## Customized Model Quantization

Please refer to [mit-han-lab/deepcompressor](https://github.com/mit-han-lab/deepcompressor/tree/main/examples/diffusion).

## Benchmark

Please refer to [app/flux/t2i/README.md](app/flux/t2i/README.md) for instructions on reproducing our paper's quality results and benchmarking inference latency on FLUX.1 models.

## Roadmap

Please check [here](https://github.com/mit-han-lab/nunchaku/issues/201) for the roadmap for March.

## Citation

If you find `nunchaku` useful or relevant to your research, please cite our paper:

```bibtex
@inproceedings{
  li2024svdquant,
  title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
  author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Related Projects

* [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://arxiv.org/abs/2211.02048), NeurIPS 2022 & T-PAMI 2023
* [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICML 2023
* [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304), ICCV 2023
* [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024
* [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481), CVPR 2024
* [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532), MLSys 2025
* [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/abs/2410.10629), ICLR 2025

## Contact Us
For enterprises interested in adopting SVDQuant or Nunchaku, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at muyangli@mit.edu.

## Acknowledgments

We thank MIT-IBM Watson AI Lab, MIT and Amazon Science Hub, MIT AI Hardware Program, National Science Foundation, Packard Foundation, Dell, LG, Hyundai, and Samsung for supporting this research. We thank NVIDIA for donating the DGX server.

We use [img2img-turbo](https://github.com/GaParmar/img2img-turbo) to train the sketch-to-image LoRA. Our text-to-image and sketch-to-image UI is built upon [playground-v.25](https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py) and [img2img-turbo](https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py), respectively. Our safety checker is borrowed from [hart](https://github.com/mit-han-lab/hart).

Nunchaku is also inspired by many open-source libraries, including (but not limited to) [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), [QServe](https://github.com/mit-han-lab/qserve), [AWQ](https://github.com/mit-han-lab/llm-awq), [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), and [Atom](https://github.com/efeslab/Atom). 