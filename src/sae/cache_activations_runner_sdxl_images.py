import io
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from diffusers.utils.import_utils import is_xformers_available

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import gc

import torch
import torchvision.transforms as transforms
from accelerate.utils import gather_object
from datasets import Array2D, Dataset, Features, Value, load_dataset
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from PIL import Image
from tqdm import tqdm

from src.hooked_model.utils import randn_tensor
from src.sae.config import CacheActivationsImagesRunnerConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

TORCH_STRING_DTYPE_MAP = {torch.float16: "float16", torch.float32: "float32"}


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class ImageDataset(Dataset):
    def __init__(self, paths, image_size: int):
        self.paths = paths
        self.image_size = image_size
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load the image from the dataset
        path = self.paths[idx]
        raw_image = Image.open(path).convert("RGB")
        # Preprocess the image into a tensor
        processed_image = self.preprocess(raw_image)

        # Return the processed image
        return processed_image


class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsImagesRunnerConfig, model, accelerator):
        self.cfg = cfg
        self.accelerator = accelerator
        self.model = model
        # hacky way to prevent initializing those objects when using only load_and_push_to_hub()
        if self.cfg.hook_names is not None:
            if is_xformers_available():
                print("Enabling xFormers memory efficient attention")
                self.model.model.enable_xformers_memory_efficient_attention()
            # self.model.model.to(self.accelerator.device)
            self.features_dict = {hookpoint: None for hookpoint in self.cfg.hook_names}
            self.scheduler = self.model.scheduler

            # Prepare timesteps
            self.scheduler.set_timesteps(self.cfg.num_inference_steps, device="cpu")
            self.scheduler_timesteps = self.scheduler.timesteps

            ds_hf = load_dataset("phiyodr/coco2017")
            ds_hf = ds_hf.shuffle(self.cfg.seed)
            paths = [
                os.path.join(self.cfg.dataset_path, example["file_name"])
                for example in ds_hf[cfg.split]
            ]
            self.file_names = [example["file_name"] for example in ds_hf[cfg.split]]
            self.dataset = ImageDataset(paths, self.cfg.height)
            if limit := self.cfg.max_num_examples:
                self.dataset = torch.utils.data.Subset(self.dataset, range(limit))
                self.file_names = self.file_names[:limit]
            self.num_examples = len(self.dataset)
            print(f"Loaded {self.num_examples} examples")
            self.indices_dataloader = self.get_batches(
                list(range(self.num_examples)), self.cfg.batch_size
            )
            self.n_buffers = len(self.indices_dataloader)
    @staticmethod
    def get_batches(items, batch_size):
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches = []

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(items))
            batch = items[start_index:end_index]
            batches.append(batch)

        return batches

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data.

        Each of the shards must be of the same format, aka the full dataset must be able to
        be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in sorted(source_dir.iterdir())]
        )

        ```

        Sharded dataset format:
        ```
        source_dir/
            shard_00000/
                dataset_info.json
                state.json
                data-00000-of-00002.arrow
                data-00001-of-00002.arrow
            shard_00001/
                dataset_info.json
                state.json
                data-00000-of-00001.arrow
        ```

        And flattens them into the format:

        ```
        output_dir/
            dataset_info.json
            state.json
            data-00000-of-00003.arrow
            data-00001-of-00003.arrow
            data-00002-of-00003.arrow
        ```

        allowing the dataset to be loaded like so:

        ```
        ds = datasets.load_from_disk(output_dir)
        ```

        Args:
            source_dir: Directory containing the sharded datasets
            output_dir: Directory to consolidate the shards into
            copy_files: If True, copy files; if False, move them and delete source_dir
        """
        first_shard_dir_name = "shard_00000"  # shard_{i:05d}

        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(p for p in output_dir.iterdir() if not p.name == ".tmp_shards")
        )
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Move dataset_info.json from any shard (all the same)
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,  # temporary
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # fingerprint is generated from dataset.__getstate__ (not including _fingerprint)
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(output_dir)

    @torch.no_grad()
    def _create_shard(
        self,
        buffer: torch.Tensor,  # buffer shape: "bs num_inference_steps+1 d_sample_size d_in",
        hook_name: str,
        input_file_names: list[str],
    ) -> Dataset:
        batch_size, n_steps, d_sample_size, d_in = buffer.shape

        # Filter buffer based on every N steps
        buffer = buffer[:, :: self.cfg.cache_every_n_timesteps, :, :]

        activations = buffer.reshape(-1, d_sample_size, d_in)
        timesteps = self.scheduler_timesteps[
            :: self.cfg.cache_every_n_timesteps
        ].repeat(batch_size)
        # Repeat each file name for each timestep
        repeated_file_names = []
        for fname in input_file_names:
            repeated_file_names.extend(
                [fname] * (n_steps // self.cfg.cache_every_n_timesteps)
            )

        shard = Dataset.from_dict(
            {
                "activations": activations,
                "timestep": timesteps,
                "file_name": repeated_file_names,
            },
            features=self.features_dict[hook_name],
        )
        return shard

    def create_dataset_feature(self, hook_name, d_in, d_out):
        self.features_dict[hook_name] = Features(
            {
                "activations": Array2D(
                    shape=(
                        d_in,
                        d_out,
                    ),
                    dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype],
                ),
                "timestep": Value(dtype="uint16"),
                "file_name": Value(dtype="string"),
            }
        )

    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:
        ### Paths setup
        assert self.cfg.new_cached_activations_path is not None

        final_cached_activation_paths = {
            n: Path(os.path.join(self.cfg.new_cached_activations_path, n))
            for n in self.cfg.hook_names
        }

        if self.accelerator.is_main_process:
            for path in final_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)
                if any(path.iterdir()):
                    raise Exception(
                        f"Activations directory ({path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
                    )

            tmp_cached_activation_paths = {
                n: path / ".tmp_shards/"
                for n, path in final_cached_activation_paths.items()
            }
            for path in tmp_cached_activation_paths.values():
                path.mkdir(exist_ok=False, parents=False)

        self.accelerator.wait_for_everyone()

        ### Create temporary sharded datasets
        if self.accelerator.is_main_process:
            print(f"Started caching {self.num_examples} activations")

        noise = randn_tensor(
            shape=(1, 4, self.cfg.height // 8, self.cfg.width // 8),
            generator=torch.Generator("cpu").manual_seed(self.cfg.seed),
            device="cpu",
            dtype=self.cfg.dtype,
        )
        noise = noise * self.model.scheduler.init_noise_sigma

        # get text embedding of an empty prompt
        with torch.no_grad():
            self.model.pipe.text_encoder.to(self.accelerator.device)
            self.model.pipe.text_encoder_2.to(self.accelerator.device)
            caption = ""
            (
                prompt_embeds,
                _,
                pooled_prompt_embeds,
                _,
            ) = self.model.encode_prompt(
                prompt=caption,
                device=self.accelerator.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            self.model.pipe.text_encoder.to("cpu")
            self.model.pipe.text_encoder_2.to("cpu")
            flush()

        for i, indices_batch in tqdm(
            enumerate(self.indices_dataloader),
            desc="Caching activations",
            total=self.n_buffers,
            disable=not self.accelerator.is_main_process,
        ):
            input_file_names = []
            with self.accelerator.split_between_processes(indices_batch) as indices_b:
                images = torch.stack([self.dataset[i] for i in indices_b], dim=0).to(
                    self.accelerator.device
                )
                input_file_names.extend([self.file_names[i] for i in indices_b])
                self.model.vae.to(self.accelerator.device)
                latents = self.model._preprocess_latents(images)
                self.model.vae.to("cpu")
                flush()
                self.model.model.to(self.accelerator.device)
                acts_cache = self.model.run_with_cache(
                    latents=latents,
                    noise=noise.repeat(len(images), 1, 1, 1).to(
                        self.accelerator.device
                    ),
                    output_type="latent",
                    num_inference_steps=self.cfg.num_inference_steps,
                    positions_to_cache=self.cfg.hook_names,
                    device=self.accelerator.device,
                    height=self.cfg.height,
                    width=self.cfg.width,
                    generator=torch.Generator(
                        device=self.accelerator.device
                    ).manual_seed(self.cfg.seed),
                    unconditional=self.cfg.unconditional,
                    prompt_embeds=prompt_embeds.repeat(len(images), 1, 1),
                    pooled_prompt_embeds=pooled_prompt_embeds.repeat(len(images), 1),
                )
                self.model.model.to("cpu")
                flush()

            self.accelerator.wait_for_everyone()
            input_file_names = gather_object(input_file_names)

            # Gather and process each hook's activations separately
            gathered_buffer = {}
            for hook_name in self.cfg.hook_names:
                gathered_buffer[hook_name] = acts_cache["output"][hook_name].cpu()
            gathered_buffer = gather_object([gathered_buffer])  # list of dicts

            if self.accelerator.is_main_process:
                for hook_name in self.cfg.hook_names:
                    gathered_buffer_acts = torch.cat(
                        [
                            gathered_buffer[i][hook_name]
                            for i in range(len(gathered_buffer))
                        ],
                        dim=0,
                    )

                    # Apply average pooling if pool is True
                    if self.cfg.pool:
                        # Original shape: [batch_size, seq_len, spatial_dim, embed_dim]
                        batch_size, seq_len, spatial_dim, embed_dim = (
                            gathered_buffer_acts.shape
                        )

                        # Calculate spatial dimensions (assuming square spatial dimension)
                        spatial_size = int(spatial_dim**0.5)  # e.g., 1024 -> 32

                        # Reshape to expose 2D spatial structure
                        reshaped = gathered_buffer_acts.view(
                            batch_size, seq_len, spatial_size, spatial_size, embed_dim
                        )

                        # Perform pooling on spatial dimensions
                        # Rearrange to [batch*seq_len*embed_dim, 1, spatial_size, spatial_size]
                        # to use avg_pool2d on the spatial dimensions only
                        pooling_shape = (
                            batch_size * seq_len * embed_dim,
                            1,
                            spatial_size,
                            spatial_size,
                        )
                        reshaped_for_pooling = reshaped.permute(0, 1, 4, 2, 3).reshape(
                            pooling_shape
                        )

                        # Apply pooling
                        pooled = torch.nn.functional.avg_pool2d(
                            reshaped_for_pooling, kernel_size=2, stride=2
                        )

                        # New spatial size after pooling
                        pooled_spatial_size = spatial_size // 2  # 16 if original was 32

                        # Reshape back to [batch_size, seq_len, embed_dim, pooled_h, pooled_w]
                        pooled = pooled.view(
                            batch_size,
                            seq_len,
                            embed_dim,
                            pooled_spatial_size,
                            pooled_spatial_size,
                        )

                        # Permute back to [batch_size, seq_len, pooled_h, pooled_w, embed_dim]
                        pooled = pooled.permute(0, 1, 3, 4, 2)

                        # Final reshape to match expected format [batch_size, seq_len, pooled_spatial_dim, embed_dim]
                        gathered_buffer_acts = pooled.reshape(
                            batch_size,
                            seq_len,
                            pooled_spatial_size * pooled_spatial_size,
                            embed_dim,
                        )

                    if self.features_dict[hook_name] is None:
                        self.create_dataset_feature(
                            hook_name,
                            gathered_buffer_acts.shape[-2],
                            gathered_buffer_acts.shape[-1],
                        )

                    print(f"{hook_name=} {gathered_buffer_acts.shape=}")

                    shard = self._create_shard(
                        gathered_buffer_acts, hook_name, input_file_names
                    )

                    shard.save_to_disk(
                        f"{tmp_cached_activation_paths[hook_name]}/shard_{i:05d}",
                        num_shards=1,
                    )
                    del gathered_buffer_acts, shard
                del gathered_buffer

        ### Concat sharded datasets together, shuffle and push to hub
        datasets = {}

        if self.accelerator.is_main_process:
            for hook_name, path in tmp_cached_activation_paths.items():
                datasets[hook_name] = self._consolidate_shards(
                    path, final_cached_activation_paths[hook_name], copy_files=False
                )
                print(f"Consolidated the dataset for hook {hook_name}")

            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                for hook_name, dataset in datasets.items():
                    dataset.push_to_hub(
                        repo_id=f"{self.cfg.hf_repo_id}_{hook_name}",
                        num_shards=self.cfg.hf_num_shards or self.n_buffers,
                        private=self.cfg.hf_is_private_repo,
                        revision=self.cfg.hf_revision,
                    )

                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)

                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )

        return datasets

    def load_and_push_to_hub(self) -> None:
        """Load dataset from disk and push it to the hub."""
        assert self.cfg.new_cached_activations_path is not None
        dataset = Dataset.load_from_disk(self.cfg.new_cached_activations_path)
        if self.accelerator.is_main_process:
            print("Loaded dataset from disk")

            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                dataset.push_to_hub(
                    repo_id=self.cfg.hf_repo_id,
                    num_shards=self.cfg.hf_num_shards
                    or (len(dataset) // self.cfg.batch_size),
                    private=self.cfg.hf_is_private_repo,
                    revision=self.cfg.hf_revision,
                )

                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)

                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )
