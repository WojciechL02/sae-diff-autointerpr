#!/bin/bash
sae_latent_idx=374

set -e

mkdir -p logs/
eval "$(conda shell.bash hook)"
conda activate zzsn
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# CUDA
module load CUDA/12.4.0
echo $CUDA_HOME

python3 scripts/sae_textual_inversion_sdxl.py \
    --pretrained_model_name_or_path stabilityai/sdxl-turbo \
    --sae_checkpoint_path /net/tscratch/people/plgwlapacz/projects/sae-diff-autointerpr/checkpoints/batch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.03125_output_249_output \
    --sae_hookpoint down_blocks.2 \
    --sae_latent_idx $sae_latent_idx \
    --cached_activations_path /net/tscratch/people/plgwlapacz/projects/sae-diff-autointerpr/activations/activations/sdxl-turbo/steps4 \
    --topk_examples 16 \
    --timestep 249 \
    --coco_dataset_path /net/pr2/projects/plgrid/plggzzsn2025/coco \
    --resolution 1024 \
    --learnable_property concept \
    --placeholder_token "<sae-concept>" \
    --initializer_token concept \
    --num_vectors 1 \
    --sae_activation_loss l1 \
    --sae_activation_loss_weight 1.0 \
    --diffusion_loss_weight 0.0 \
    --sae_max_feature_act 2.43 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 2000 \
    --learning_rate 1e-04 \
    --scale_lr \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --save_steps 100 \
    --output_dir sae_textual_inversion_sdxl/sae_latent_idx${sae_latent_idx} \
    --report_to wandb \
    --seed 42
