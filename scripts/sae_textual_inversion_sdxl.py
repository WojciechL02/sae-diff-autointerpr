#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# From: https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion_sdxl.py

import argparse
import gc
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List
import json

import diffusers
import einops
import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import Dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

sys.path.append(os.path.dirname(__file__))

from src.hooked_model.utils import locate_block, retrieve
from src.sae.sae import Sae

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.34.0.dev0")

logger = get_logger(__name__)


def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "textual_inversion",
    ]

    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    unet,
    vae,
    args,
    accelerator,
    weight_dtype,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder_1),
        text_encoder_2=accelerator.unwrap_model(text_encoder_2),
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )  # TODO: DDPMScheduler
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        None
        if args.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )
    images = []
    for _ in range(args.num_validation_images):
        image = pipeline(
            args.validation_prompt, num_inference_steps=25, generator=generator
        ).images[
            0
        ]  # TODO: num_inference_steps
        images.append(image)

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    tracker_key: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()
    return images


def save_progress(
    text_encoder,
    placeholder_token_ids,
    accelerator,
    args,
    save_path,
    safe_serialization=True,
    tokenizer=None,
    step=None,
):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )

    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if tokenizer is not None:
        other_ids = [i for i in range(len(tokenizer)) if i != placeholder_token_ids]
        embeddings = (
            accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight[other_ids]
        )
        tokens, scores = get_most_similar_tokens(
            learned_embeds.detach(), embeddings.detach(), tokenizer, k=8
        )
        table = wandb.Table(columns=["token", "cosine_similarity"])
        for tok, score in zip(tokens, scores):
            table.add_data(tok, score)

        table = {f"top_tokens_table_{step}": table}
        accelerator.log(table, step=step)
        accelerator.log(
            {"max_token_cos_sim": scores[1]}
        )  # first is just a learned token

    if safe_serialization:
        safetensors.torch.save_file(
            learned_embeds_dict, save_path, metadata={"format": "pt"}
        )
    else:
        torch.save(learned_embeds_dict, save_path)


def find_topk_activating_examples(activations_per_sample, latent_idx, k=10):
    topk_indices = torch.argsort(
        activations_per_sample[:, latent_idx], dim=0, descending=True
    )[:k]
    return topk_indices


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def run_with_cache(
    unet: torch.nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    positions_to_cache: List[str],
    save_input: bool = False,
    save_output: bool = True,
    **kwargs,
):
    """
    Run UNet while caching intermediate values at specified positions.
    Returns both the final image and a dictionary of cached values.
    """
    cache_input, cache_output = (
        dict() if save_input else None,
        dict() if save_output else None,
    )
    hooks = [
        _register_cache_hook(
            unet,
            position,
            cache_input,
            cache_output,
        )
        for position in positions_to_cache
    ]
    hooks = [hook for hook in hooks if hook is not None]

    image = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        **kwargs,
    ).sample

    # Stack cached tensors along time dimension
    cache_dict = {}
    if save_input:
        for position, block in cache_input.items():
            cache_input[position] = torch.stack(block, dim=1)
        cache_dict["input"] = cache_input

    if save_output:
        for position, block in cache_output.items():
            cache_output[position] = torch.stack(block, dim=1)
        cache_dict["output"] = cache_output

    for hook in hooks:
        hook.remove()

    return image, cache_dict


def _register_cache_hook(
    model: torch.nn.Module,
    position: str,
    cache_input: Dict,
    cache_output: Dict,
    unconditional: bool = False,
    pool: bool = False,
    do_classifier_free_guidance: bool = False,
):
    block = locate_block(position, model)

    def hook(module, input, kwargs, output):
        if cache_input is not None:
            if position not in cache_input:
                cache_input[position] = []
            input_to_cache = retrieve(input, unconditional, do_classifier_free_guidance)
            if len(input_to_cache.shape) == 4:
                input_to_cache = input_to_cache.view(
                    input_to_cache.shape[0], input_to_cache.shape[1], -1
                ).permute(0, 2, 1)
            if pool:
                input_to_cache = input_to_cache.mean(dim=1, keepdim=True)
            cache_input[position].append(input_to_cache)

        if cache_output is not None:
            if position not in cache_output:
                cache_output[position] = []
            output_to_cache = retrieve(
                output, unconditional, do_classifier_free_guidance
            )
            if len(output_to_cache.shape) == 4:
                output_to_cache = output_to_cache.view(
                    output_to_cache.shape[0], output_to_cache.shape[1], -1
                ).permute(0, 2, 1)
            if pool:
                output_to_cache = output_to_cache.mean(dim=1, keepdim=True)
            cache_output[position].append(output_to_cache)

    return block.register_forward_hook(hook, with_kwargs=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as initializer word.",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="object",
        help="Choose between 'object' and 'style'",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="How many times to repeat the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--sae_checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint of the SAE model.",
    )
    parser.add_argument(
        "--sae_hookpoint",
        type=str,
        required=True,
        help=(
            "UNet block name to register the forward hook for the SAE latents e.g. down_blocks.2."
        ),
    )
    parser.add_argument(
        "--sae_latent_idx",
        type=int,
        default=0,
        help=("SAE latent index to perform the textual inversion on."),
    )
    parser.add_argument(
        "--cached_activations_path",
        type=str,
        required=True,
        help=(
            "Path to the cached UNet activations at `sae_hookpoint`. The activations are used to retrieve samples used for textual inversion."
        ),
    )
    parser.add_argument(
        "--coco_dataset_path",
        type=str,
        required=True,
        help=(
            "Path to the COCO dataset. The dataset is used to retrieve samples used for textual inversion."
        ),
    )
    parser.add_argument(
        "--topk_examples",
        type=int,
        default=5,
        help=(
            "Number of examples that activate `sae_latent_idx` the most, as well as the number of examples used for textual inversion.."
        ),
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=249,
        help="Diffusion timestep used for SAE activations.",
    )
    parser.add_argument(
        "--sae_activation_loss",
        type=str,
        default="l2",
        help="Regularization loss used to enforce maximization of SAE activations at `sae_latent_idx`.",
    )
    parser.add_argument(
        "--sae_activation_loss_weight",
        type=float,
        default=1.0,
        help="Regulariazation strength for the SAE activation loss.",
    )
    parser.add_argument(
        "--diffusion_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the diffusion loss term.",
    )
    parser.add_argument(
        "--sae_max_feature_act",
        type=float,
        required=True,
        help="Maximal activation value that learned feature can have.",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

# TODO: improve templates e.g. 'a photo containing a concept of a {}'
sae_concept_templates_small = [
    "an image where {} is visible",
    "a photo containing a concept of a {}",
    "a scene with a visual concept of {}",
    "a rendering that includes the concept of a {}",
    "a composition with a strong presence of {}",
    "an image featuring {}-like visual elements",
    "a photo where the {} concept is present",
    "a surface with a concept of {} texture",
    "an artwork inspired by the concept of a {}",
]


class SAETextualInversionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        tokenizer_1,
        tokenizer_2,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        file_names=None,
    ):
        self.data_root = data_root
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        if file_names is not None:
            self.image_paths = [
                os.path.join(self.data_root, file_name) for file_name in file_names
            ]
        else:
            self.image_paths = [
                os.path.join(self.data_root, file_name)
                for file_name in os.listdir(self.data_root)
            ]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = {
            "object": imagenet_templates_small,
            "style": imagenet_style_templates_small,
            "concept": imagenet_templates_small,
        }[learnable_property]
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.crop = (
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        )

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        example["image_id"] = i % self.num_images
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["original_size"] = (image.height, image.width)

        image = image.resize((self.size, self.size), resample=self.interpolation)

        if self.center_crop:
            y1 = max(0, int(round((image.height - self.size) / 2.0)))
            x1 = max(0, int(round((image.width - self.size) / 2.0)))
            image = self.crop(image)
        else:
            y1, x1, h, w = self.crop.get_params(image, (self.size, self.size))
            image = transforms.functional.crop(image, y1, x1, h, w)

        example["crop_top_left"] = (y1, x1)

        example["input_ids_1"] = self.tokenizer_1(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_2"] = self.tokenizer_2(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def get_most_similar_tokens(learned_embeds, embeddings, tokenizer, k=8):
    learned_embeds = F.normalize(learned_embeds, dim=1)
    embeddings = F.normalize(embeddings, dim=1)
    similarities = torch.matmul(embeddings, learned_embeds.T)
    values, indices = similarities.topk(k=k, dim=0)
    tokens = tokenizer.convert_ids_to_tokens(indices.T.squeeze())
    return tokens, values


def get_embedding_info(text_encoder, placeholder_token_ids):
    learned_embed_grad_norm = (
        text_encoder.text_model.embeddings.token_embedding.weight.grad.detach()[
            min(placeholder_token_ids) : max(placeholder_token_ids) + 1
        ]
        .data.norm(2)
        .item()
    )
    learned_embed_norm = (
        text_encoder.text_model.embeddings.token_embedding.weight[
            min(placeholder_token_ids) : max(placeholder_token_ids) + 1
        ]
        .detach()
        .data.norm(2)
        .item()
    )
    return learned_embed_grad_norm, learned_embed_norm


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load SAE
    sae = Sae.load_from_disk(os.path.join(args.sae_checkpoint_path, args.sae_hookpoint))
    sae.requires_grad_(False)
    sae.to(accelerator.device, dtype=weight_dtype)

    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )

    # Add the placeholder token in tokenizer_1
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(
            f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}"
        )

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer_1.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    num_added_tokens = tokenizer_2.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The 2nd tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer_1.encode(args.initializer_token, add_special_tokens=False)
    token_ids_2 = tokenizer_2.encode(args.initializer_token, add_special_tokens=False)

    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1 or len(token_ids_2) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer_1.convert_tokens_to_ids(placeholder_tokens)
    initializer_token_id_2 = token_ids_2[0]
    placeholder_token_ids_2 = tokenizer_2.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder_1.resize_token_embeddings(len(tokenizer_1))
    text_encoder_2.resize_token_embeddings(len(tokenizer_2))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder_1.get_input_embeddings().weight.data
    token_embeds_2 = text_encoder_2.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
        for token_id in placeholder_token_ids_2:
            token_embeds_2[token_id] = token_embeds_2[initializer_token_id_2].clone()

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder_1.text_model.encoder.requires_grad_(False)
    text_encoder_1.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_1.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder_2.text_model.encoder.requires_grad_(False)
    text_encoder_2.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_2.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        text_encoder_1.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        # only optimize the embeddings
        [
            text_encoder_1.text_model.embeddings.token_embedding.weight,
            text_encoder_2.text_model.embeddings.token_embedding.weight,
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    placeholder_token = " ".join(
        tokenizer_1.convert_ids_to_tokens(placeholder_token_ids)
    )

    # Load cached activations dataset
    cached_activations_dataset = Dataset.load_from_disk(
        os.path.join(args.cached_activations_path, args.sae_hookpoint),
        keep_in_memory=False,
    )
    cached_activations_dataset.set_format(
        type="torch", columns=["activations", "timestep", "file_name"], dtype=sae.dtype
    )
    cached_activations_dataset = cached_activations_dataset.filter(
        lambda x: x["timestep"] == args.timestep, batched=True
    )

    # Retrieve most activating examples
    avg_activations_per_sample = torch.zeros(
        (len(cached_activations_dataset), sae.num_latents), dtype=sae.dtype
    )
    batch_size = 64
    cached_activations_dataloader = torch.utils.data.DataLoader(
        cached_activations_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(cached_activations_dataloader),
            total=len(cached_activations_dataloader),
            desc="Retriving max activating examples",
        ):
            acts = batch["activations"].to(sae.device)
            acts = einops.rearrange(
                acts,
                "batch sample_size d_model -> (batch sample_size) d_model",
            )
            out = sae.pre_acts(acts)
            # Reshape to get per-sample activations and compute mean for each sample
            out = out.view(
                batch["activations"].shape[0], -1, sae.num_latents
            )  # [batch, sample_size, num_latents]
            batch_avg_activations = out.mean(dim=1).to(
                dtype=sae.dtype
            )  # [batch, num_latents]

            # Store in the correct indices
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(cached_activations_dataset))
            avg_activations_per_sample[start_idx:end_idx] = batch_avg_activations

    topk_indices = find_topk_activating_examples(
        avg_activations_per_sample, args.sae_latent_idx, k=args.topk_examples
    )
    topk_examples = cached_activations_dataset[topk_indices.tolist()]
    topk_examples_file_names = topk_examples["file_name"]

    del cached_activations_dataset, cached_activations_dataloader
    flush()

    with torch.no_grad():
        topk_activations = topk_examples["activations"].to(sae.device)
        topk_activations = einops.rearrange(
            topk_activations,
            "batch sample_size d_model -> (batch sample_size) d_model",
        )
        topk_sae_latents = sae.pre_acts(topk_activations)
        topk_sae_latents = topk_sae_latents.view(
            args.topk_examples, -1, sae.num_latents
        )
        topk_active_positions_mask = topk_sae_latents[:, :, args.sae_latent_idx] > 0.0
        topk_active_positions_mask = topk_active_positions_mask.float()

        del topk_activations, topk_sae_latents
        flush()

    # Dataset and DataLoaders creation:
    train_dataset = SAETextualInversionDataset(
        data_root=args.coco_dataset_path,
        file_names=topk_examples_file_names,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        placeholder_token=placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    text_encoder_1.train()
    text_encoder_2.train()
    # Prepare everything with our `accelerator`.
    text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler
        )
    )

    # Move vae and unet and text_encoder_2 to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            os.getenv("WANDB_PROJECT", "textual_inversion"), config=vars(args)
        )

        # Save arguments to a file
        save_path = os.path.join(args.output_dir, "config.json")
        with open(save_path, "w") as f:
            json.dump(vars(args), f, indent=4)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # keep original embeddings as reference
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder_1)
        .get_input_embeddings()
        .weight.data.clone()
    )
    orig_embeds_params_2 = (
        accelerator.unwrap_model(text_encoder_2)
        .get_input_embeddings()
        .weight.data.clone()
    )

    training_step = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder_1.train()
        text_encoder_2.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([text_encoder_1, text_encoder_2]):
                # Convert images to latent space
                latents = (
                    vae.encode(batch["pixel_values"].to(dtype=weight_dtype))
                    .latent_dist.sample()
                    .detach()
                )
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                # timesteps = timesteps.long()
                timesteps = torch.full(
                    (bsz,), args.timestep, device=latents.device, dtype=torch.long
                )

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states_1 = (
                    text_encoder_1(batch["input_ids_1"], output_hidden_states=True)
                    .hidden_states[-2]
                    .to(dtype=weight_dtype)
                )
                encoder_output_2 = text_encoder_2(
                    batch["input_ids_2"], output_hidden_states=True
                )
                encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(
                    dtype=weight_dtype
                )
                original_size = [
                    (
                        batch["original_size"][0][i].item(),
                        batch["original_size"][1][i].item(),
                    )
                    for i in range(args.train_batch_size)
                ]
                crop_top_left = [
                    (
                        batch["crop_top_left"][0][i].item(),
                        batch["crop_top_left"][1][i].item(),
                    )
                    for i in range(args.train_batch_size)
                ]
                target_size = (args.resolution, args.resolution)
                add_time_ids = torch.cat(
                    [
                        torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                        for i in range(args.train_batch_size)
                    ]
                ).to(accelerator.device, dtype=weight_dtype)
                added_cond_kwargs = {
                    "text_embeds": encoder_output_2[0],
                    "time_ids": add_time_ids,
                }
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states_1, encoder_hidden_states_2], dim=-1
                )

                # Predict the noise residual and cache intermediate activations
                model_pred, acts_cache = run_with_cache(
                    unet,
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    positions_to_cache=[args.sae_hookpoint],
                    added_cond_kwargs=added_cond_kwargs,
                )

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                diffusion_loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )

                # Compute SAE latent activations
                acts = acts_cache["output"][
                    args.sae_hookpoint
                ]  # (batch_size, num_timesteps, sample_size, d_model)
                acts = acts.squeeze(1)  # NOTE: only one timestep is used
                acts = einops.rearrange(
                    acts,
                    "batch_size sample_size d_model -> (batch_size sample_size) d_model",
                )
                sae_latent_acts = sae.pre_acts(acts)
                sae_latent_acts_org = sae_latent_acts.view(bsz, -1, sae.num_latents)

                # Mask out positions without concept
                active_positions_mask = topk_active_positions_mask[batch["image_id"]]
                sae_latent_acts_org = (
                    active_positions_mask.unsqueeze(-1) * sae_latent_acts_org
                )
                sae_latent_acts = torch.clamp(
                    sae_latent_acts_org[:, :, args.sae_latent_idx],
                    max=args.sae_max_feature_act,
                )

                other_features_indices = [
                    i for i in range(sae.num_latents) if i != args.sae_latent_idx
                ]
                other_features = sae_latent_acts_org[:, :, other_features_indices]

                # Compute the SAE activation loss
                if args.sae_activation_loss == "l2":
                    sae_loss1 = -torch.mean(sae_latent_acts**2)
                    sae_loss2 = torch.mean(other_features**2)
                elif args.sae_activation_loss == "l1":
                    sae_loss1 = -torch.mean(sae_latent_acts.abs())
                    sae_loss2 = torch.mean(other_features.abs())
                else:
                    raise ValueError(
                        f"Unknown SAE activation loss {args.sae_activation_loss}"
                    )

                loss = (
                    args.diffusion_loss_weight * diffusion_loss
                    + args.sae_activation_loss_weight * (sae_loss1 + sae_loss2)
                )
                accelerator.backward(loss)

                #### LOG LEARNED EMBEDDING INFO ####
                grad_norm1, embed_norm1 = get_embedding_info(
                    text_encoder_1, placeholder_token_ids
                )
                grad_norm2, embed_norm2 = get_embedding_info(
                    text_encoder_1, placeholder_token_ids
                )
                accelerator.log(
                    {"learned_embed_grad_norm1": grad_norm1}, step=training_step
                )
                accelerator.log(
                    {"learned_embed_norm1": embed_norm1}, step=training_step
                )
                accelerator.log(
                    {"learned_embed_grad_norm2": grad_norm2}, step=training_step
                )
                accelerator.log(
                    {"learned_embed_norm2": embed_norm2}, step=training_step
                )
                training_step += 1
                #####################################

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer_1),), dtype=torch.bool)
                index_no_updates[
                    min(placeholder_token_ids) : max(placeholder_token_ids) + 1
                ] = False
                index_no_updates_2 = torch.ones((len(tokenizer_2),), dtype=torch.bool)
                index_no_updates_2[
                    min(placeholder_token_ids_2) : max(placeholder_token_ids_2) + 1
                ] = False

                with torch.no_grad():
                    accelerator.unwrap_model(
                        text_encoder_1
                    ).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[
                        index_no_updates
                    ]
                    accelerator.unwrap_model(
                        text_encoder_2
                    ).get_input_embeddings().weight[
                        index_no_updates_2
                    ] = orig_embeds_params_2[
                        index_no_updates_2
                    ]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = f"learned_embeds-steps-{global_step}.safetensors"
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        text_encoder_1,
                        placeholder_token_ids,
                        accelerator,
                        args,
                        save_path,
                        True,
                        tokenizer_1,
                        global_step,
                    )
                    weight_name = f"learned_embeds_2-steps-{global_step}.safetensors"
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        text_encoder_2,
                        placeholder_token_ids_2,
                        accelerator,
                        args,
                        save_path,
                        safe_serialization=True,
                    )

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if (
                        args.validation_prompt is not None
                        and global_step % args.validation_steps == 0
                    ):
                        images = log_validation(
                            text_encoder_1,
                            text_encoder_2,
                            tokenizer_1,
                            tokenizer_2,
                            unet,
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            epoch,
                        )

            logs = {
                "diffusion_loss": diffusion_loss.detach().item(),
                "sae_loss1": sae_loss1.detach().item(),
                "sae_loss2": sae_loss2.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.validation_prompt:
            images = log_validation(
                text_encoder_1,
                text_encoder_2,
                tokenizer_1,
                tokenizer_2,
                unet,
                vae,
                args,
                accelerator,
                weight_dtype,
                epoch,
                is_final_validation=True,
            )

        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warning(
                "Enabling full model saving because --push_to_hub=True was specified."
            )
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder_1),
                text_encoder_2=accelerator.unwrap_model(text_encoder_2),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer_1,
                tokenizer_2=tokenizer_2,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        weight_name = "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder_1,
            placeholder_token_ids,
            accelerator,
            args,
            save_path,
            safe_serialization=True,
        )
        weight_name = "learned_embeds_2.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder_2,
            placeholder_token_ids_2,
            accelerator,
            args,
            save_path,
            safe_serialization=True,
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
