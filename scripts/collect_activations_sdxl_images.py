"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""

import os
import sys

from simple_parsing import parse

sys.path.append(os.path.dirname(__file__))
from accelerate import Accelerator
from diffusers import (
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

from src.hooked_model.hooked_model_sdxl_images import HookedDiffusionModel
from src.sae.cache_activations_runner_sdxl_images import CacheActivationsRunner
from src.sae.config import CacheActivationsImagesRunnerConfig


def run():
    args = parse(CacheActivationsImagesRunnerConfig)
    accelerator = Accelerator()
    # define model
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet").to(
        dtype=args.dtype
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_name,
        unet=unet,
        torch_dtype=args.dtype,
        use_safetensors=True,
    )
    scheduler = DDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    scheduler.set_timesteps(args.num_inference_steps)
    print(scheduler.timesteps)
    hooked_model = HookedDiffusionModel(
        model=unet,
        scheduler=scheduler,
        encode_prompt=pipe.encode_prompt,
        vae=pipe.vae,
        pipe=pipe,
    )

    CacheActivationsRunner(args, hooked_model, accelerator).run()


if __name__ == "__main__":
    run()
