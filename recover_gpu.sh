#!/usr/bin/env bash
# Attempt to recover NVIDIA compute after a CUDA launch/Xid failure without rebooting.
#
# This is intentionally non-interactive by default. If sudo needs a password, the
# script prints the manual commands instead of hanging inside train_iterate.sh.

set -u

GPU_ID="${GPU_ID:-0}"

echo "GPU recovery: checking NVIDIA state for GPU $GPU_ID..."
nvidia-smi || {
  echo "GPU recovery: nvidia-smi is not responding; a full driver reload or reboot may be required." >&2
  exit 1
}

echo
echo "GPU recovery: recent NVIDIA kernel faults:"
(dmesg --ctime 2>/dev/null || journalctl -k --since "2 hours ago" --no-pager 2>/dev/null) \
  | grep -Ei "NVRM|Xid|CUDA|nvidia" \
  | tail -n 12 || true

echo
echo "GPU recovery: trying nvidia-smi GPU reset..."
if nvidia-smi --gpu-reset -i "$GPU_ID"; then
  echo "GPU recovery: nvidia-smi reset completed."
  nvidia-smi || true
  exit 0
fi

if command -v sudo >/dev/null 2>&1; then
  echo
  echo "GPU recovery: trying sudo -n nvidia-smi GPU reset..."
  if sudo -n nvidia-smi --gpu-reset -i "$GPU_ID"; then
    echo "GPU recovery: sudo nvidia-smi reset completed."
    nvidia-smi || true
    exit 0
  fi

  echo
  echo "GPU recovery: trying sudo -n nvidia_uvm reload..."
  if sudo -n modprobe -r nvidia_uvm && sudo -n modprobe nvidia_uvm; then
    echo "GPU recovery: nvidia_uvm reload completed."
    nvidia-smi || true
    exit 0
  fi
fi

cat >&2 <<EOF
GPU recovery: automatic reset was not available.

Manual no-reboot recovery options:
  sudo nvidia-smi --gpu-reset -i $GPU_ID
  sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm

If those report the GPU is in use, inspect users with:
  sudo fuser -v /dev/nvidia*

Then stop only stale compute clients and retry the reset/reload. If Xorg or the
display stack owns the device, a full GPU reset may be blocked, but reloading
nvidia_uvm is often enough to clear compute-only CUDA fallout.
EOF

exit 1
