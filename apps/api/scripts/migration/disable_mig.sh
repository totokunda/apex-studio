#!/bin/bash
set -e

echo "Deleting MIG compute instances..."
sudo nvidia-smi mig -dci -i 0 || true

echo "Deleting MIG GPU instances..."
sudo nvidia-smi mig -dgi -i 0 || true

echo "Disabling MIG mode..."
sudo nvidia-smi -i 0 -mig 0

echo "Done. Current GPU layout:"
nvidia-smi -L
