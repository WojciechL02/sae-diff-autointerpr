sbatch slurm_scripts/submit_job.sh scripts/collect_activations_sdxl_images.py \
    --hook_names down_blocks.2 \
    --dataset_path /net/pr2/projects/plgrid/plggzzsn2025/coco \
    --split validation \
    --model_name stabilityai/sdxl-turbo \
    --num_inference_steps 4 \
    --unconditional True
