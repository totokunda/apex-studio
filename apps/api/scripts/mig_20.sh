#!/bin/bash
set -e

echo "Enabling MIG on GPU 0..."
sudo nvidia-smi -i 0 -mig 1

echo "Deleting existing MIG instances..."
sudo nvidia-smi mig -dci -i 0 || true
sudo nvidia-smi mig -dgi -i 0 || true

echo "Creating 2g.20gb GPU instance..."
sudo nvidia-smi mig -cgi 2g.20gb -i 0 -C

echo "Resulting devices:"
nvidia-smi -L
