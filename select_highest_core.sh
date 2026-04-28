#!/bin/bash

target_dir="models/core"

# Extract highest number, then find the directory that contains it
highest_ev=$(ls "$target_dir" | grep -oP '\d+(?=ev)' | sort -rn | head -n 1)
highest_path=$(find "$target_dir" -maxdepth 1 -type d -name "*${highest_ev}ev*" | head -n 1)

echo "$highest_path"
