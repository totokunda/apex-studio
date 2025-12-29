watch -n 1 "
echo '=== MIG Layout ==='
nvidia-smi -L
echo
echo '=== Active Compute Apps ==='
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader | column -t
"
