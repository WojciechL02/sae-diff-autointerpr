for lr in 1e-3 1e-2 1e-1 1 10 100; do
    for wd in 0.0 1e-5 1e-3 1e-1; do
        bash run_sae_textual_inversion_sdxl_add_to_prompt.sh $lr $wd
    done
done