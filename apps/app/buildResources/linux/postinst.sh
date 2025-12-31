#!/bin/bash
# Post-installation script for Apex Studio on Linux

set -e

# Create necessary directories
mkdir -p ~/.apex-studio/cache
mkdir -p ~/.apex-studio/components
mkdir -p ~/.apex-studio/logs

# Set up desktop integration
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database -q /usr/share/applications || true
fi

# Update icon cache
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
fi

# Check for NVIDIA drivers and display helpful message
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Apex Studio will use CUDA for acceleration."
elif [ -d /opt/rocm ]; then
    echo "AMD ROCm detected. Apex Studio will use ROCm for acceleration."
else
    echo "No GPU acceleration detected. Apex Studio will run in CPU mode."
    echo "For best performance, install NVIDIA drivers with CUDA support."
fi

echo "Apex Studio installation complete!"
echo "You can start the application from your applications menu or by running 'apex-studio'."

