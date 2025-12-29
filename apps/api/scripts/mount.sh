#!/bin/bash
set -e

DEVICE="/dev/nvme1n1"
MOUNT_POINT="/mnt/localssd"
FSTAB="/etc/fstab"

# 1) Format the device if it isnâ€™t already ext4
if ! sudo blkid "$DEVICE" | grep -q ext4; then
  sudo mkfs.ext4 -F "$DEVICE"
fi

# 2) Ensure the mount point exists
sudo mkdir -p "$MOUNT_POINT"

# 3) Mount right now (wonâ€™t block boot if missing)
sudo mount -o discard,defaults,nofail,x-systemd.device-timeout=10s \
     "$DEVICE" "$MOUNT_POINT" || true

# 4) Fix ownership so your user can write
sudo chown "$USER:$USER" "$MOUNT_POINT"

# 5) Patch /etc/fstab:
#    - If thereâ€™s already a line starting with "$DEVICE ", update its ext4 options
#    - Otherwise append a new, correct line
if grep -q "^$DEVICE[[:space:]]" "$FSTAB"; then
  sudo sed -i -E "\|^$DEVICE[[:space:]]| s|ext4[[:space:]]|ext4 discard,defaults,nofail,x-systemd.device-timeout=10s |" \
         "$FSTAB"
else
  echo "$DEVICE $MOUNT_POINT ext4 discard,defaults,nofail,x-systemd.device-timeout=10s 0 0" \
    | sudo tee -a "$FSTAB" >/dev/null
fi

# 6) Reload systemd so mounts take effect immediately (and non-fatal)
sudo systemctl daemon-reload
sudo systemctl restart local-fs.target || true

echo "âœ… Local SSD ($DEVICE) mounted (or skipped) and /etc/fstab updated with nofail."