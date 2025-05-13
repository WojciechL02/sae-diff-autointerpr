import os
import sys

sys.path.append(os.path.dirname(__file__))
import argparse

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from tqdm import tqdm, trange

from src.sae.sae import Sae

# configure matplotlib
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26
plt.rc("font", size=SMALL_SIZE, family="Times New Roman")  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE, labelsize=SMALL_SIZE)  # f


def parse_args():
    parser = argparse.ArgumentParser(description="Find and visualize max activating examples for each SAE neuron")
    parser.add_argument("--sae_ckpt_path", type=str, required=True, help="Path to the SAE checkpoint")
    parser.add_argument("--sae_hookpoint", type=str, required=True, help="Hookpoint to use for the SAE")
    parser.add_argument(
        "--cached_activations_path", type=str, required=True, help="Path to the cached activations dataset"
    )
    parser.add_argument(
        "--timestep", type=int, default=249, help="Activations from this diffusion timestep will be used"
    )
    parser.add_argument("--coco_dataset_path", type=str, required=True, help="Path to the COCO dataset")
    parser.add_argument("--topk_examples", type=int, default=5, help="Number of top activating examples to visualize")
    parser.add_argument(
        "--output_dir", type=str, default="max_activating_examples", help="Directory to save the output images"
    )
    return parser.parse_args()

def run(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16
    torch.set_grad_enabled(False)

    sae = Sae.load_from_disk(
        os.path.join(
            args.sae_ckpt_path,
            args.sae_hookpoint,
        ),
        device=device,
    ).to(dtype)

    activations_dataset = Dataset.load_from_disk(
        os.path.join(args.cached_activations_path, args.sae_hookpoint), keep_in_memory=False
    )
    activations_dataset.set_format(type="torch", columns=["activations", "timestep", "file_name"], dtype=dtype)

    # filter dataset to only include activations from timestep
    activations_dataset = activations_dataset.filter(lambda x: x["timestep"] == args.timestep, batched=True)
    print(f"Number of samples in dataset: {len(activations_dataset)}")

    avg_activations_per_sample = torch.zeros((len(activations_dataset), sae.num_latents), dtype=dtype)

    batch_size = 16
    dl = torch.utils.data.DataLoader(activations_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            acts = batch["activations"].to(sae.device)
            acts = einops.rearrange(
                acts,
                "batch sample_size d_model -> (batch sample_size) d_model",
            )
            out = sae.pre_acts(acts)
            # Reshape to get per-sample activations and compute mean for each sample
            out = out.view(batch["activations"].shape[0], -1, sae.num_latents)  # [batch, sample_size, num_latents]
            batch_avg_activations = out.mean(dim=1).to(dtype=dtype)  # [batch, num_latents]

            # Store in the correct indices
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(activations_dataset))
            avg_activations_per_sample[start_idx:end_idx] = batch_avg_activations

    def find_topk_activating_examples(activations_per_sample, latent_idx, k=10):
        topk_indices = torch.argsort(activations_per_sample[:, latent_idx], dim=0, descending=True)[:k]
        return topk_indices

    for latent_idx in trange(sae.num_latents, desc="Plotting max activating examples for latent neurons"):
        topk_indices = find_topk_activating_examples(
            avg_activations_per_sample, latent_idx, args.topk_examples
        )  # find topk samples containing patches with higest activations
        topk_samples = activations_dataset[topk_indices.tolist()]
        sae_latents = []
        activations = topk_samples["activations"].to(sae.device)
        timesteps = topk_samples["timestep"]
        file_names_topk = topk_samples["file_name"]
        activations = einops.rearrange(
            activations,
            "batch sample_size d_model -> (batch sample_size) d_model",
        )
        out = sae.pre_acts(activations)
        sae_latents = out.view(args.topk_examples, -1, sae.num_latents)

        fig, axes = plt.subplots(2, len(topk_indices), figsize=(18, 6))

        # Plot max activating examples in two rows:
        # Row 1: Original images
        # Row 2: Model activations
        for i in range(len(topk_indices)):
            # Model 1 images
            img = Image.open(os.path.join(args.coco_dataset_path, file_names_topk[i]))
            img = img.resize((512, 512))
            img = img.convert("RGB")

            # Process activations for model 1
            sae_latent_activations = sae_latents[i].reshape(
                int(torch.sqrt(torch.tensor(sae_latents.shape[1])).item()),
                int(torch.sqrt(torch.tensor(sae_latents.shape[1])).item()),
                -1,
            )[:, :, latent_idx]
            # Convert latent activations to numpy and normalize
            activation_map = sae_latent_activations[:, :].detach().cpu().numpy()
            activation_map = (activation_map - activation_map.min()) / (
                activation_map.max() - activation_map.min() + 1e-8
            )

            # Calculate upscale factor to match image size for model 1
            patch_size = 512 // activation_map.shape[0]
            activation_map = np.kron(activation_map, np.ones((patch_size, patch_size)))

            # Create heatmap overlays
            heatmap = np.uint8(plt.cm.jet(activation_map)[..., :3] * 255)
            heatmap = Image.fromarray(heatmap)

            # Blend original images with heatmaps
            blended_img = Image.blend(img, heatmap, alpha=0.4)

            # Calculate average activation for the image
            avg_activation = sae_latent_activations.mean().item()

            # Row 1: Original images
            axes[0, i].imshow(img)
            axes[0, i].axis("off")
            axes[0, i].set_title(
                f"Activation: {avg_activation:.2f}\nTimestep: {int(timesteps[i].item())}",
                fontsize=SMALL_SIZE,
            )
            if i == 0:
                axes[0, 0].set_ylabel("Original Images", fontsize=SMALL_SIZE)

            # Row 2: Activations
            axes[1, i].imshow(blended_img)
            axes[1, i].axis("off")
            if i == 0:
                axes[1, 0].set_ylabel("Activations", fontsize=SMALL_SIZE)

        plt.suptitle(f"Max Activating Examples for Neuron {latent_idx}", fontsize=BIGGER_SIZE)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"max_activating_examples_{latent_idx}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    run(parse_args())
