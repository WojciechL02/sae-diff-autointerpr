import os

os.chdir("../")

import sys

sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# Path to your saved embeddings
embeddings_path = "sae_textual_inversion_sdxl/sae_latent_idx123/learned_embeds_2-steps-400.safetensors"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
with safe_open(embeddings_path, framework="pt") as f:
    # Get all the tensors in the file
    tensor_names = f.keys()
    if placeholder_token in tensor_names:
        learned_embeds = f.get_tensor(placeholder_token).to(device)


embeddings = text_encoder_2.get_input_embeddings().weight.to(device)

learned_embeds = F.normalize(learned_embeds, dim=1)
embeddings = F.normalize(embeddings, dim=1)
similarities = torch.matmul(embeddings, learned_embeds.T)

values, indices = similarities.topk(k=16, dim=0)

# print()
tokens = tokenizer_2.convert_ids_to_tokens(indices.T.squeeze())
print("Cosine similarity:")
print(f"Tokens: {tokens}")
print(values.T.squeeze())
print()
