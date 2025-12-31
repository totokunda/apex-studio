#!/bin/bash
# Post-removal script for Apex Studio on Linux

set -e

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database -q /usr/share/applications || true
fi

# Update icon cache
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
fi

echo "Apex Studio has been removed."
echo "Note: User data in ~/.apex-studio has been preserved."
echo "To completely remove all data, run: rm -rf ~/.apex-studio"

