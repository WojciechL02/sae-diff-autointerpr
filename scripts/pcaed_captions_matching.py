import os
import sys

sys.path.append(os.path.dirname(__file__))
import argparse

import einops
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from diffusers import DiffusionPipeline
from tqdm import tqdm, trange

from src.pca import PCA
from src.sae.sae import Sae
from src.vocab import get_vocabulary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrive the most similar concepts based on the 1st PCA direction of captions retrived from the topk activating examples"
    )
    parser.add_argument("--sae_ckpt_path", type=str, required=True, help="Path to the SAE checkpoint")
    parser.add_argument("--sae_hookpoint", type=str, required=True, help="Hookpoint to use for the SAE")
    parser.add_argument(
        "--cached_activations_path", type=str, required=True, help="Path to the cached activations dataset"
    )
    parser.add_argument(
        "--timestep", type=int, default=249, help="Activations from this diffusion timestep will be used"
    )
    parser.add_argument("--topk_examples", type=int, default=5, help="Number of top activating examples to visualize")
    parser.add_argument(
        "--output_dir", type=str, default="pcaed_captions_matching", help="Directory to save the output images"
    )
    parser.add_argument("--vocab_name", type=str, default="laion", help="Vocabulary name")
    parser.add_argument("--vocab_size", type=int, default=-1, help="Vocabulary size")
    parser.add_argument("--concept_embeddings_path", type=str, default=None, help="Path to the concept embeddings")
    parser.add_argument("--topk_concepts", type=int, default=5, help="Number of top concepts to retrieve")
    return parser.parse_args()


def run(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16
    torch.set_grad_enabled(False)

    # load SDXL text encoder
    pipe_sd_turbo = DiffusionPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    text_encoder = pipe_sd_turbo.encode_prompt

    # get text embeddings for concepts in the vocabulary
    vocab = get_vocabulary(args.vocab_name, args.vocab_size)

    concept_embeddings_path = (
        os.path.join(
            args.output_dir,
            f"{args.vocab_name}_{args.vocab_size if args.vocab_size > 0 else 'all'}_concept_embeddings.pt",
        )
        if args.concept_embeddings_path is None
        else args.concept_embeddings_path
    )

    def get_concept_embeddings(text_encoder, vocab: list[str], device="cuda"):
        concepts = []

        for concept in tqdm(vocab, desc="Getting concept embeddings", total=len(vocab)):
            with torch.no_grad():
                prompt_embeds, _, pooled_prompt_embeds, _ = text_encoder(concept, device=device)
                concept_embedding = pooled_prompt_embeds  # NOTE: we use pooled embeddings from OpenCLIP
            concepts.append(concept_embedding)

        concepts = torch.stack(concepts).squeeze()
        # concepts = F.normalize(torch.stack(concepts).squeeze(), dim=1)
        # concepts = F.normalize(concepts-torch.mean(concepts, dim=0), dim=1)
        return concepts

    if os.path.exists(concept_embeddings_path):
        print(f"Loading concept embeddings from {concept_embeddings_path}")
        concept_embeddings = torch.load(concept_embeddings_path, map_location="cpu").to(device)
    else:
        concept_embeddings = get_concept_embeddings(text_encoder, vocab, device=device)
        torch.save(concept_embeddings.cpu(), concept_embeddings_path)
        print(f"Saved concept embeddings to {concept_embeddings_path}")

    print(f"Concept embeddings shape: {concept_embeddings.shape}")

    # load SAE
    sae = Sae.load_from_disk(
        os.path.join(
            args.sae_ckpt_path,
            args.sae_hookpoint,
        ),
        device=device,
    ).to(dtype)

    # load cached activations
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
        for i, batch in tqdm(enumerate(dl), total=len(dl), desc="Computing average activations"):
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

    coco_dataset = load_dataset("phiyodr/coco2017")

    with open(os.path.join(args.output_dir, "concepts.tsv"), "w") as f:
        for latent_idx in trange(sae.num_latents, desc="Matching captions for latent neurons"):
            if latent_idx == 0:
                f.write("latent_idx\tconcepts\n")

            topk_indices = find_topk_activating_examples(
                avg_activations_per_sample, latent_idx, args.topk_examples
            )  # find topk samples containing patches with higest activations
            topk_samples = activations_dataset[topk_indices.tolist()]
            file_names_topk = topk_samples["file_name"]

            # filter coco dataset to only include the topk samples
            coco_topk_samples = coco_dataset["validation"].filter(lambda x: x["file_name"] in file_names_topk)
            # concatenate all caption for each image
            topk_samples_captions = [" ".join(captions) for captions in coco_topk_samples["captions"]]

            # get text embeddings for topk samples captions
            prompt_embeds, _, pooled_prompt_embeds, _ = text_encoder(topk_samples_captions, device=device)
            topk_samples_caption_embeddings = pooled_prompt_embeds  # NOTE: we use pooled embeddings from OpenCLIP

            # perform PCA on the text embeddings and extract the first PC direction
            pca = PCA(n_components=1).to(topk_samples_caption_embeddings.device).fit(topk_samples_caption_embeddings.float())
            pca_captions_embedding = (pca.components_.sum(dim=1, keepdim=True) + pca.mean_).mean(1)

            similarities = F.cosine_similarity(
                pca_captions_embedding.expand_as(concept_embeddings), concept_embeddings, dim=1
            )
            most_similar_indices = torch.argsort(similarities, descending=True)[: args.topk_concepts]
            most_similar_indices = most_similar_indices.cpu().numpy()
            most_similar_concepts = [vocab[i] for i in most_similar_indices]

            if latent_idx % 100 == 0:
                print(f"Latent {latent_idx}: {', '.join(most_similar_concepts)}")

            # save to file
            f.write(f"{latent_idx}\t{','.join(most_similar_concepts)}\n")


if __name__ == "__main__":
    run(parse_args())
