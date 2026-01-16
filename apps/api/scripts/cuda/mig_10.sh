#!/bin/bash
set -e

echo "Enabling MIG on GPU 0..."
sudo nvidia-smi -i 0 -mig 1

echo "Deleting existing MIG instances..."
sudo nvidia-smi mig -dci -i 0 || true
sudo nvidia-smi mig -dgi -i 0 || true

echo "Creating 1g.10gb GPU instance (10GB VRAM profile)..."
sudo nvidia-smi mig -cgi 1g.10gb -i 0 -C

echo "Resulting devices:"
nvidia-smi -L


