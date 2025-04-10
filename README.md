# SAE for Diffusion Models

Repo for SAE training on activations of diffusion models.

I tried to make the code as general as possible, so that it can be used for different architectures, no matter if the model is implemented in `diffusers` or not.
However, for sure certain models will need adaptations. I hope it will be a relatively good starting point.

## Overview of implementation
Training SAEs and gathering the data are two completely separate processes. Given prepared dataset in huggingface format, SAE training should be straightforward and the same for each model and modality.

Some models are compatible with `diffusers` and some are not.
The main interface of the repo is `HookedDiffusionModel` class, which is a wrapper around a diffusion model.
I tried making it very general so that is can be used for any diffusion model, no matter if it is implemented in `diffusers` or not and no matter if it is a UNet-based model or a Transformer-based model.

### How to adapt your model
You need to wrap your model in `HookedDiffusionModel` class.

**model** - Your denoiser, either UNet or Transformer. It needs to output predicted noise for each timestep.

**scheduler** - Noise scheduler. If possible, use `src/hooked_model/scheduler.py`, which contains DDIM scheduler adapted from `diffusers`.
If not possible, your custom scheduler needs to implement `step()` and `scale_model_input()` methods. Additionally it needs to have certain attributes (look at `src/hooked_model/hooked_model.py` for more details).

**vae** - VAE model for latent space. `None` if you do not use the latent model.

**encode_prompt** - Function that encodes prompt into embeddings.
```python
def encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
    ...
    return prompt_embeds, negative_prompt_embeds
```

**get_timesteps** - Function that returns discrete timesteps based on the number of inference steps and the scheduler.
```python
def get_timesteps(timesteps, device):
    ...
    return timesteps
```


### Inference
Implemented in `__call__` method.
Basically 3 steps:
1. Encode prompt
2. Prepare latents and timesteps
3. Run denoising loop

It is general and adapted to SD v1.5, but should be easily adaptable to other models.


## Setup 🚀

```bash
pip install -r requirements.txt
```


## Steps to train SAEs on activations of diffusion models
Training SAEs on activations of diffision models can be split into 3 steps:

1. Find a layer to apply SAE on.
2. Gather activations of the model on the dataset from the chosen layer.
3. Train SAEs on the activations.

### Finding a layer

Given a model's capability that we want to interpret, you can simply look and measure how much each layer contributes to the model's capability.

A good starting point is to skip one layer at a time and see how it impacts the final generation.

Notebook `notebooks/ablate_blocks.ipynb` contains a simple example of how to do this.

### Gathering activations
Once you have a layer to apply SAE on, you need to gather activations of the model on the dataset.

**Dataset**
You need some dataset of prompts of whatever. You need to define it in the constructor of `CacheActivationsRunner` class.

Define your model in `src/scripts/collect_activations.py` and run the script.
```bash
python src/scripts/collect_activations.py --hook_names <hook_name>
```

For all run arguments, see `CacheActivationsRunnerConfig` in `src/sae/config.py`.

### Training SAEs
```bash
python src/scripts/train.py --dataset_path <path_to_dataset>
```

For all run arguments, see `TrainConfig` in `src/sae/config.py`.
