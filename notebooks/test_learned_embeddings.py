import os

os.chdir("../")

import sys

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def get_most_similar_tokens(learned_embeds, embeddings, tokenizer):
    learned_embeds = F.normalize(learned_embeds, dim=1)
    embeddings = F.normalize(embeddings, dim=1)
    similarities = torch.matmul(embeddings, learned_embeds.T)

    values, indices = similarities.topk(k=16, dim=0)

    # print()
    tokens = tokenizer.convert_ids_to_tokens(indices.T.squeeze())
    print("Cosine similarity:")
    print(f"Tokens: {tokens}")
    print(values.T.squeeze())
    print()


# Path to your saved embeddings
feature = 374
step = 1000
lr = 10.0
embeddings_path1 = f"sae_textual_inversion_sdxl/sae_latent_idx{feature}_lr{lr}/learned_embeds-steps-{step}.safetensors"
embeddings_path2 = f"sae_textual_inversion_sdxl/sae_latent_idx{feature}/learned_embeds_2-steps-{step}.safetensors"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_encoder_1 = CLIPTextModel.from_pretrained(
    "stabilityai/sdxl-turbo",
    subfolder="text_encoder",
)
tokenizer_1 = CLIPTokenizer.from_pretrained(
    "stabilityai/sdxl-turbo", subfolder="tokenizer"
)

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    "stabilityai/sdxl-turbo",
    subfolder="text_encoder_2",
)
tokenizer_2 = CLIPTokenizer.from_pretrained(
    "stabilityai/sdxl-turbo", subfolder="tokenizer_2"
)

# The placeholder token you used during training
placeholder_token = "<sae-concept>"

# Load the learned embeddings from safetensors file
with safe_open(embeddings_path1, framework="pt") as f:
    # Get all the tensors in the file
    tensor_names = f.keys()
    if placeholder_token in tensor_names:
        learned_embeds1 = f.get_tensor(placeholder_token).to(device)


embeddings1 = text_encoder_1.get_input_embeddings().weight.to(device)

with safe_open(embeddings_path2, framework="pt") as f:
    # Get all the tensors in the file
    tensor_names = f.keys()
    if placeholder_token in tensor_names:
        learned_embeds2 = f.get_tensor(placeholder_token).to(device)


embeddings2 = text_encoder_2.get_input_embeddings().weight.to(device)

get_most_similar_tokens(learned_embeds1, embeddings1, tokenizer_1)
get_most_similar_tokens(learned_embeds2, embeddings2, tokenizer_2)
