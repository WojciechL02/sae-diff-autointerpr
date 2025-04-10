import inspect
from typing import Callable, Dict, List, Optional, Union

import torch

from src.hooked_model.utils import (
    locate_block,
    postprocess_image,
    randn_tensor,
    retrieve,
)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class HookedDiffusionModel:
    def __init__(
        self,
        model: torch.nn.Module,
        scheduler,
        encode_prompt: Callable,
        vae: Optional[torch.nn.Module] = None,
        pipe: Optional[torch.nn.Module] = None,
    ):
        """
        Initialize a hooked diffusion model.

        Args:
            model (torch.nn.Module): The base diffusion model (UNet or Transformer)
            scheduler: The noise scheduler
            encode_prompt (Callable): Function to encode text prompts into embeddings
            get_timesteps (Callable): Function to generate timesteps for inference
            vae (torch.nn.Module, optional): The VAE model for latent encoding/decoding
        """
        # Core components
        self.model = model
        self.scheduler = scheduler
        self.vae = vae
        self.encode_prompt = encode_prompt
        self.pipe = pipe

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noise: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        self.scheduler.num_inference_steps = num_inference_steps
        batch_size = latents.shape[0]
        # Get timesteps for the diffusion process
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = 1280
        add_time_ids = self._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        # Run denoising process
        # latents = self._preprocess_latents(images)
        self._denoise_loop(
            latents,
            timesteps,
            noise,
            prompt_embeds,
            add_text_embeds,
            add_time_ids,
            extra_step_kwargs,
        )
        # image = self._postprocess_latents(latents, output_type, generator)
        # return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.model.config.addition_time_embed_dim * len(add_time_ids)
            + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.model.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.no_grad()
    def run_with_hooks(
        self,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        """
        Run the pipeline with hooks at specified positions.

        Args:
            position_hook_dict: Dictionary mapping model positions to hooks.
                Keys: Position strings indicating where to register hooks
                Values: Single hook function or list of hook functions
                Each hook should accept (module, input, output) arguments
            device: Device to run inference on
            **kwargs: Additional arguments passed to base pipeline
        """
        hooks = []
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]

        try:
            image = self(
                device=device,
                **kwargs,
            )
        finally:
            for hook in hooks:
                hook.remove()

        return image

    @torch.no_grad()
    def run_with_cache(
        self,
        positions_to_cache: List[str],
        save_input: bool = False,
        save_output: bool = True,
        unconditional: bool = False,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        """
        Run pipeline while caching intermediate values at specified positions.
        Compatible with both UNet and Transformer-based models.

        Returns both the final image and a dictionary of cached values.
        """
        cache_input, cache_output = (
            dict() if save_input else None,
            dict() if save_output else None,
        )
        hooks = [
            self._register_cache_hook(
                position,
                cache_input,
                cache_output,
                unconditional,
                do_classifier_free_guidance=False,
            )
            for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]

        self(
            device=device,
            **kwargs,
        )

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

        return cache_dict

    def _register_cache_hook(
        self,
        position: str,
        cache_input: Dict,
        cache_output: Dict,
        unconditional: bool = False,
        pool: bool = False,
        do_classifier_free_guidance: bool = True,
    ):
        block = locate_block(position, self.model)

        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                input_to_cache = retrieve(
                    input, unconditional, do_classifier_free_guidance
                )
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

    def _register_general_hook(self, position, hook):
        block = locate_block(position, self.model)
        return block.register_forward_hook(hook)

    def _denoise_loop(
        self,
        latents,
        timesteps,
        noise,
        prompt_embeds,
        add_text_embeds,
        add_time_ids,
        extra_step_kwargs,
    ):
        timestep_cond = None
        for i, t in enumerate(timesteps):
            t_ = t.repeat(latents.shape[0], 1)
            latent_model_input = self.scheduler.add_noise(latents, noise, t_)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            latent_model_input = latent_model_input.to(self.model.dtype)
            _ = self.model(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # latents_ = self.scheduler.step(
            #     noise_pred,
            #     t,
            #     latent_model_input,
            #     **extra_step_kwargs,
            #     return_dict=False,
            # )[0]

        # return latents

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = True
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    def _preprocess_latents(self, latents):
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(
                next(iter(self.vae.post_quant_conv.parameters())).dtype
            )
        if latents.dtype != self.vae.dtype:
            latents = latents.to(self.vae.dtype)
        latents = self.vae.encode(latents).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.to(self.model.dtype)

        return latents

    def _postprocess_latents(self, latents, output_type, generator):
        if not output_type == "latent" and self.vae is not None:
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )
            has_latents_mean = (
                hasattr(self.vae.config, "latents_mean")
                and self.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(self.vae.config, "latents_std")
                and self.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / self.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(
                latents,
                return_dict=False,
            )[0]
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        image = postprocess_image(image, output_type=output_type)

        if output_type == "latent":
            image = image.cpu().numpy()
        return image

    def _prepare_latents(
        self,
        batch_size,
        num_images_per_prompt,
        in_channels,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (batch_size * num_images_per_prompt, in_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
