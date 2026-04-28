#!/usr/bin/env bash

# Set path relative to project root
TARGET_DIR="models/core"
KEEP_FILE="manifest.json"  # File to exclude from deletion

# 1. Identify the King of the Hill (Highest EV)
MAX_EV=$(ls -1 "$TARGET_DIR" 2>/dev/null | grep -oP '\d+(?=ev)' | sort -rn | head -n 1 || true)

if [[ -z "$MAX_EV" ]]; then
    echo "Error: No directory with 'XXXev' pattern found in $TARGET_DIR"
    exit 1
fi

# 2. Get the specific directory name (handles ties by picking the first)
KEEP_DIR=$(ls -1 "$TARGET_DIR" | grep "${MAX_EV}ev" | head -n 1)

echo "--- Core Cleanup ---"
echo "Highest EV found: $MAX_EV"
echo "KEEPING: $KEEP_DIR"
echo "KEEPING: $KEEP_FILE"
echo "--------------------"

# 3. Logic to delete everything else
# We loop through everything in the folder
for item in "$TARGET_DIR"/*; do
    basename_item=$(basename "$item")

    # Skip if it's the one we want to keep
    if [[ "$basename_item" == "$KEEP_DIR" || "$basename_item" == "$KEEP_FILE" ]]; then
        continue
    fi

    # Check for --force flag to actually delete
    if [[ "${1:-}" == "--force" ]]; then
        echo "Deleting: $basename_item"
        rm -rf "$item"
    else
        echo "[DRY RUN] Would delete: $basename_item"
    fi
done

if [[ "${1:-}" != "--force" ]]; then
    echo ""
    echo "!!! This was a DRY RUN. No files were deleted. !!!"
    echo "Run: ./core_clean.sh --force  to execute cleanup."
fi
