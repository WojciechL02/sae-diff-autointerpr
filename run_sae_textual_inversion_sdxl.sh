sae_latent_idx=123

sbatch slurm_scripts/submit_job.sh scripts/sae_textual_inversion_sdxl.py \
    --pretrained_model_name_or_path stabilityai/sdxl-turbo \
    --sae_checkpoint_path /net/tscratch/people/plgpiorczynskim/sae-diff-autointerpr/checkpoints/coco2017/sdxl-turbo/batch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.03125_output_249_output \
    --sae_hookpoint down_blocks.2 \
    --sae_latent_idx $sae_latent_idx \
    --cached_activations_path /net/tscratch/people/plgpiorczynskim/sae-diff-autointerpr/activations/coco2017/sdxl-turbo/steps4 \
    --topk_examples 5 \
    --timestep 249 \
    --coco_dataset_path /net/pr2/projects/plgrid/plggzzsn2025/coco \
    --resolution 1024 \
    --learnable_property concept \
    --placeholder_token "<sae-concept>" \
    --initializer_token concept \
    --num_vectors 1 \
    --sae_activation_loss l2 \
    --sae_activation_loss_weight 1e-4 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 500 \
    --learning_rate 5e-04 \
    --scale_lr \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --save_steps 100 \
    --output_dir sae_textual_inversion_sdxl/sae_latent_idx${sae_latent_idx} \
    --report_to wandb \
    --seed 42
