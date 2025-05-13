bash slurm_scripts/submit_job.sh scripts/max_activating_examples.py \
    --sae_ckpt_path checkpoints/coco2017/sdxl-turbo/batch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.03125_output_249_output \
    --sae_hookpoint down_blocks.2 \
    --cached_activations_path activations/coco2017/sdxl-turbo/steps4 \
    --timestep 249 \
    --topk_examples 5 \
    --output_dir pcaed_captions_matching \
    --vocab_name laion_bigrams \
    --topk_concepts 10



