"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""

import os
import sys

from simple_parsing import parse

sys.path.append(os.path.dirname(__file__))
from accelerate import Accelerator
from diffusers import DiffusionPipeline

from src.hooked_model.hooked_model_sdxl import HookedDiffusionModel
from src.sae.cache_activations_runner_sdxl import CacheActivationsRunner
from src.sae.config import CacheActivationsRunnerConfig


def run():
    args = parse(CacheActivationsRunnerConfig)
    accelerator = Accelerator()
    # define model
    pipe = DiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=args.dtype,
        variant="fp16",
    ).to(accelerator.device)
    model = pipe.unet
    scheduler = pipe.scheduler
    hooked_model = HookedDiffusionModel(
        model=model,
        scheduler=scheduler,
        encode_prompt=pipe.encode_prompt,
        vae=pipe.vae,
    )

    CacheActivationsRunner(args, hooked_model, accelerator).run()


if __name__ == "__main__":
    run()
