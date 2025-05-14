bash slurm_scripts/submit_job.sh scripts/max_activating_examples.py \
    --sae_checkpoint_path checkpoints/coco2017/sdxl-turbo/batch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.03125_output_249_output \
    --sae_hookpoint down_blocks.2 \
    --cached_activations_path activations/coco2017/sdxl-turbo/steps4 \
    --timestep 249 \
    --coco_dataset_path data/coco2017 \
    --topk_examples 5 \
    --output_dir max_activating_examples \



