#!/bin/bash
# Author: Mohammad Al-Jarrah
# Download the demonstration data sets released with Diffusion Policy (Chi et al., 2023).
#
#   bash scripts/download_data.sh pusht     # Push-T: data/pusht/pusht_cchi_v7_replay.zarr   (~30 MB)
#   bash scripts/download_data.sh square    # Square: data/square/ph/low_dim_abs.hdf5         (~50 MB;
#                                           #   the archive holds all robomimic tasks, 1.8 GB)
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://diffusion-policy.cs.columbia.edu/data/training

case "${1:-}" in
  pusht)
    mkdir -p data/pusht
    curl -L -o data/pusht.zip "$URL/pusht.zip"
    unzip -q -o data/pusht.zip -d data/                    # -> data/pusht/pusht_cchi_v7_replay.zarr
    rm data/pusht.zip
    ls -d data/pusht/pusht_cchi_v7_replay.zarr ;;
  square)
    mkdir -p data/square/ph
    curl -L -o data/robomimic_lowdim.zip "$URL/robomimic_lowdim.zip"
    unzip -q -o -j data/robomimic_lowdim.zip 'robomimic/datasets/square/ph/low_dim_abs.hdf5' -d data/square/ph/
    rm data/robomimic_lowdim.zip
    ls data/square/ph/low_dim_abs.hdf5 ;;
  *)
    echo "usage: bash scripts/download_data.sh pusht|square"; exit 1 ;;
esac
