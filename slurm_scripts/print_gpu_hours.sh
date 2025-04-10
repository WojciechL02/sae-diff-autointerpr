echo "GPU hours used: $(hpc-jobs-history -A plgzzsn2025-gpu-a100 -d 7 | awk '$11 ~ /^[0-9.]*$/ {sum += $11} END {print sum}')"
