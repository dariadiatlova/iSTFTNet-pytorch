#!/bin/bash
cur="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)" # to make running the script from anywhere available
cd $cur
mkdir src/data
cd src/data
# download awesome_checkpoints.zip (generator weights, discriminator weights, onnx models)
gdown https://drive.google.com/uc?id=1rewyoyuB4QtHL_rb6xOHpOumyU87riHP;
unzip -qq awesome_checkpoints.zip;