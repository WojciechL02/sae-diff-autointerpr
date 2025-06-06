sae_latent_idx=374
learning_rate=10



sbatch slurm_scripts/submit_job.sh scripts/sae_textual_inversion_sdxl_add_to_prompt.py \
    --pretrained_model_name_or_path stabilityai/sdxl-turbo \
    --sae_checkpoint_path /net/tscratch/people/plgpiorczynskim/sae-diff-autointerpr/checkpoints/coco2017/sdxl-turbo/batch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.03125_output_249_output \
    --sae_hookpoint down_blocks.2 \
    --sae_latent_idx $sae_latent_idx \
    --cached_activations_path /net/tscratch/people/plgpiorczynskim/sae-diff-autointerpr/activations/coco2017/sdxl-turbo/steps4 \
    --topk_examples 16 \
    --timestep 249 \
    --coco_dataset_path /net/pr2/projects/plgrid/plggzzsn2025/coco \
    --resolution 1024 \
    --concept_vocab_name laion_bigrams \
    --concept_embeddings_path /net/tscratch/people/plgpiorczynskim/sae-diff-autointerpr/sae_textual_inversion_sdxl_add_to_prompt_pooled/laion_bigrams_all_concept_embeddings.pt \
    --sae_activation_loss l1 \
    --sae_activation_loss_weight 1.0 \
    --sae_loss_max_weight 1.0 \
    --sae_loss_min_weight 0.0 \
    --diffusion_loss_weight 1.0 \
    --sae_max_feature_act 10 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 1000 \
    --learning_rate ${learning_rate} \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --save_steps 200 \
    --do_validation \
    --validation_steps 200 \
    --output_dir "sae_textual_inversion_sdxl_add_to_prompt_pooled/sae_latent_idx${sae_latent_idx}_${learning_rate}_max_only_diff_loss" \
    --report_to wandb \
    --seed 42